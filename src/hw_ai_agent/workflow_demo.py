from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from .agent import AgentError, CodingAgent, DEFAULT_SYSTEM_PROMPT
from .client import ModelClientError
from .reporting import public_report
from .policy import Policy
from .session import SessionStore
from .budget import BudgetExceeded
from .tools import ToolRuntime


FIXED_TAX = "def add_tax(amount):\n    return amount * 1.2\n"
FIXED_RECEIPT = "def receipt_total(prices):\n    return sum(prices)\n"
WORKFLOW_TESTS = '''import unittest
from tax import add_tax
from receipt import receipt_total

class CheckoutTests(unittest.TestCase):
    def test_tax(self):
        self.assertAlmostEqual(add_tax(50), 60)

    def test_receipt(self):
        self.assertEqual(receipt_total([10, 20, 30]), 60)
'''

OPTIMIZATION_TESTS = WORKFLOW_TESTS.replace("from receipt import receipt_total", "from receipt import receipt_total\nfrom checkout import total_label\nfrom formatting import format_amount") + '''
    def test_checkout(self):
        self.assertEqual(total_label([10, 20]), "36.00")

    def test_format(self):
        self.assertEqual(format_amount(12.5), "12.50")
'''

ADDITIONAL_FILES = {
    "checkout.py": "from tax import add_tax\nfrom receipt import receipt_total\nfrom formatting import format_amount\n\ndef total_label(prices):\n    return format_amount(add_tax(receipt_total(prices)))\n",
    "formatting.py": "def format_amount(amount):\n    return f'{amount:.2f}'\n",
    "tax_backup.py": "def add_tax(amount):\n    return amount + 20\n",
    "receipt_backup.py": "def receipt_total(prices):\n    return len(prices)\n",
    "scratch_preview.py": "def preview():\n    return 'sample receipt'\n",
}
OPTIONAL_FILES = ("tax_backup.py", "receipt_backup.py", "scratch_preview.py")
WORKFLOW_REQUEST = (
    "Inspect the project, fix its errors and optimize the code. "
    "You can read, create, edit or remove files and run tests using the available tools. "
    "Do not change the tests. Preserve the public API used by them. "
    "Run the tests after your changes and explain your decisions in Russian."
)
WORKFLOW_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT.replace(
    "After relevant tests pass, immediately provide your final summary without more tool calls.",
    "Inspect all source files and their module dependencies before making changes. "
    "Complete the requested fixes and optimizations before your final test run. "
    "After final tests pass, provide your summary."
)


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
                      approvals="ask", on_event=None, crash_after_write=False,
                      client=None, max_rounds=20, budget_limits=None):
    if db_path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("Session database must be outside workspace")
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "tax.py").write_text("def add_tax(amount):\n    return amount + 20\n", encoding="utf-8")
    (workspace / "receipt.py").write_text("def receipt_total(prices):\n    return len(prices)\n", encoding="utf-8")
    real_model = client is not None
    tests = OPTIMIZATION_TESTS if real_model else WORKFLOW_TESTS
    if real_model:
        for name, content in ADDITIONAL_FILES.items():
            (workspace / name).write_text(content, encoding="utf-8")
    (workspace / "test_checkout.py").write_text(tests, encoding="utf-8")
    runtime_class = CrashAfterWriteRuntime if crash_after_write else ToolRuntime
    policy = Policy.from_dict({"defaults": {"write_file": "ask", "Bash": "ask"}})
    runtime = runtime_class(workspace, policy=policy, mode=mode, approve=approve,
                            trash_dir=db_path.resolve().parent / "trash")
    baseline = runtime.bash("python -m unittest discover -v")
    if not baseline.startswith("exit_code=1\n"):
        raise ValueError("Expected failing workflow baseline")
    metadata = {"kind": "workflow", "approvals": approvals,
                "demo": {"tests": tests, "baseline": baseline,
                         "optional_files": list(OPTIONAL_FILES) if real_model else [],
                         "cleanup_required": real_model}}
    with SessionStore(db_path) as store:
        agent = CodingAgent(client if real_model else WorkflowClient(), runtime, store=store,
                            on_event=on_event, session_metadata=metadata, max_rounds=max_rounds, budget_limits=budget_limits,
                            system_prompt=WORKFLOW_SYSTEM_PROMPT if real_model else DEFAULT_SYSTEM_PROMPT)
        error = None
        try:
            agent.run(WORKFLOW_REQUEST)
        except (AgentError, BudgetExceeded, ModelClientError) as exc:
            error = str(exc)
        return verify_workflow(agent, runtime, metadata, error=error)


def verify_workflow(agent, runtime, metadata, error=None):
    output = runtime.bash("python -m unittest discover -v")
    unchanged = (runtime.root / "test_checkout.py").read_bytes() == metadata["demo"]["tests"].encode("utf-8")
    policy_events = [e for e in agent.events if e.get("event") == "policy_decision"]
    prompts = Counter(e["tool"] for e in policy_events if e.get("outcome") == "waiting")
    attempts = Counter(json.loads(e["arguments"])["path"] for e in agent.events
                       if e.get("event") == "tool_call" and e.get("name") == "write_file")
    removed = [name for name in metadata["demo"].get("optional_files", []) if not (runtime.root / name).exists()]
    cleanup = len(removed) >= 2 if metadata["demo"].get("cleanup_required") else True
    functional = output.startswith("exit_code=0\n") and unchanged
    report = {"workspace": str(runtime.root), "session_id": agent.session_id, "budget": agent.budget,
              "mode": runtime.mode, "baseline": metadata["demo"]["baseline"],
              "verification": output, "tests_unchanged": unchanged,
              "success": error is None and functional and cleanup, "functional_success": functional,
              "error": error,
              "cleanup_complete": cleanup, "removed_optional_files": removed,
              "permission_requests": dict(prompts), "write_attempts": dict(attempts),
              "events": agent.events,
              "provider": getattr(getattr(agent.client, "settings", None), "provider", "scripted"),
              "model": getattr(getattr(agent.client, "settings", None), "model", "scripted"),
              "usage": [e["usage"] for e in agent.events if e.get("event") == "model_usage"]}
    (runtime.root / "workflow-report.json").write_text(
        json.dumps(public_report(report, runtime.root), ensure_ascii=False, indent=2), encoding="utf-8")
    return report
