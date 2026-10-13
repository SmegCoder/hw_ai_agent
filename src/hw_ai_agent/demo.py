from __future__ import annotations
import difflib
import hashlib
import json
from pathlib import Path
from .agent import AgentError, CodingAgent
from .client import ModelClientError, OpenAICompatibleClient
from .budget import BudgetExceeded
from .tools import ToolRuntime
from .policy import Policy
from .session import SessionStore
from .reporting import public_report

BROKEN_ORDER = '''def calculate_total(prices, discount_percent):
    return sum(prices) - discount_percent
'''

ORDER_TESTS = '''import unittest
from order import calculate_total

class OrderTests(unittest.TestCase):
    def test_ten_percent(self):
        self.assertAlmostEqual(calculate_total([100, 50], 10), 135)
    def test_zero_discount(self):
        self.assertEqual(calculate_total([20, 30], 0), 50)
    def test_full_discount(self):
        self.assertEqual(calculate_total([20, 30], 100), 0)
    def test_empty_order(self):
        self.assertEqual(calculate_total([], 10), 0)
    def test_fractional_discount(self):
        self.assertAlmostEqual(calculate_total([40, 60], 12.5), 87.5)
    def test_invalid_discount(self):
        for value in [-1, 101]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculate_total([100], value)
'''

def initialize_demo(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "order.py").write_text(BROKEN_ORDER, encoding="utf-8")
    (workspace / "test_order.py").write_text(ORDER_TESTS, encoding="utf-8")

def run_demo(client: OpenAICompatibleClient, workspace: Path, max_rounds: int = 12,
             on_event=None, *, policy: Policy | None = None, approve=None,
             store: SessionStore | None = None, session_metadata: dict | None = None,
             mode: str = "safe", trash_dir: Path | None = None, plan_mode=False, task_mode=False, approve_plan=None, budget_limits=None, journal_mode=False) -> dict:
    initialize_demo(workspace)
    runtime = ToolRuntime(workspace, policy=policy, approve=approve, on_policy_event=on_event,
                          mode=mode, trash_dir=trash_dir)
    before = ("Baseline not run: Bash is disabled until plan approval." if plan_mode
              else runtime.bash("python -m unittest discover -v"))
    if not plan_mode and not before.startswith("exit_code=1\n"):
        raise AgentError("Demo baseline did not produce the expected failing tests")
    test = workspace / "test_order.py"
    test_hash = hashlib.sha256(test.read_bytes()).hexdigest()
    metadata = {**(session_metadata or {}), "kind": "demo", "demo": {"baseline": before, "test_hash": test_hash}}
    agent = CodingAgent(client, runtime, max_rounds=max_rounds, on_event=on_event,
                        store=store, session_metadata=metadata, plan_mode=plan_mode, task_mode=task_mode, approve_plan=approve_plan, budget_limits=budget_limits, journal_mode=journal_mode)
    error = None
    answer = ""
    try:
        answer = agent.run(
            "Read order.py and test_order.py. Fix calculate_total: discount_percent is a percentage, "
            "valid from 0 to 100 inclusive; invalid values must raise ValueError. "
            "Run tests through Bash using python -m unittest discover -v. "
            "Once they pass, stop calling tools and give a short final summary. "
            "Do not change tests or create auxiliary files. Respond in Russian."
        )
    except (AgentError, BudgetExceeded, ModelClientError) as exc:
        error = str(exc)
    return verify_demo(client, workspace, runtime, before, test_hash, answer,
                       agent.events, error, agent.session_id)

def verify_demo(client, workspace, runtime, before, test_hash, answer, events, error=None, session_id=None):
    after = runtime.bash("python -m unittest discover -v")
    test = workspace / "test_order.py"
    unchanged = test_hash == hashlib.sha256(test.read_bytes()).hexdigest()
    model_ran_tests = any(e.get("event") == "tool_result" and e.get("name") in {"Bash", "bash"}
                          and "Ran 6 tests" in e["content"] and e["content"].startswith("exit_code=0\n")
                          for e in events)
    diff = "".join(difflib.unified_diff(BROKEN_ORDER.splitlines(True),
                    (workspace / "order.py").read_text().splitlines(True),
                    fromfile="before/order.py", tofile="after/order.py"))
    report = {"provider": client.settings.provider, "model": client.settings.model,
              "workspace": str(workspace), "baseline": before, "verification": after,
              "tests_unchanged": unchanged, "model_ran_passing_tests": model_ran_tests,
              "success": error is None and unchanged and model_ran_tests and after.startswith("exit_code=0\n"),
              "error": error,
              "diff": diff, "answer": answer,
              "usage": [e["usage"] for e in events if e.get("event") == "model_usage"] or client.usage,
              "events": events,
              "session_id": session_id,
              "budget": next((e["budget"] for e in reversed(events) if e.get("event") == "budget"), None),
              "policy_events": runtime.policy_events}
    (workspace / "demo-report.json").write_text(
        json.dumps(public_report(report, workspace), ensure_ascii=False, indent=2), encoding="utf-8")
    return report
