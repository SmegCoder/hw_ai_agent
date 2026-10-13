from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .agent import CodingAgent
from .budget import BudgetExceeded, BudgetLimits
from .session import SessionStore, UncertainActionError
from .tools import ToolRuntime


def call(identifier, name, **arguments):
    return {"id": identifier, "function": {"name": name, "arguments": json.dumps(arguments)}}


class BudgetDemoClient:
    """Fixed responses test runtime decisions without a model or API key."""
    def __init__(self, *responses):
        self.responses = iter(responses)

    def complete(self, messages, tools):
        return next(self.responses)


def run_budget_demo(on_event=None):
    checks, examples = {}, {}
    write = call("write", "write_file", path="value.py", content="fixed")
    read = call("read", "read_file", path="value.py")
    listing = call("list", "list_files", path=".")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        with SessionStore(root / ".agent/state.sqlite3") as store:
            scenarios = {
                "model_requests": (BudgetLimits(model_requests=1), [{"tool_calls": [listing]}]),
                "tool_calls": (BudgetLimits(tool_calls=1), [{"tool_calls": [write, read]}]),
                "side_effects": (BudgetLimits(side_effects=0), [{"tool_calls": [listing, write]}]),
                "result_bytes": (BudgetLimits(result_bytes=0), [{"tool_calls": [write]}]),
            }
            for dimension, (limits, responses) in scenarios.items():
                workspace = root / dimension
                workspace.mkdir()
                agent = CodingAgent(BudgetDemoClient(*responses), ToolRuntime(workspace, mode="auto"),
                                    max_rounds=limits.model_requests, budget_limits=limits,
                                    store=store, on_event=on_event)
                try:
                    agent.run("Check the runtime budget")
                    checks[dimension + "_exhausted"] = False
                except BudgetExceeded as exc:
                    checks[dimension + "_exhausted"] = exc.dimension == dimension
                if dimension != "model_requests":
                    checks[dimension + "_batch_untouched"] = not any(workspace.iterdir())
                examples[dimension] = agent.budget

            workspace = root / "resume"
            workspace.mkdir()
            def pause(event):
                if on_event:
                    on_event(event)
                if event["event"] == "tool_result":
                    raise KeyboardInterrupt
            first = CodingAgent(BudgetDemoClient({"tool_calls": [listing, write, read]}),
                                ToolRuntime(workspace, mode="auto"), store=store, on_event=pause)
            try:
                first.run("Inspect, write and read")
            except KeyboardInterrupt:
                pass
            before = first.budget
            resumed = CodingAgent(BudgetDemoClient({"content": "Done"}), ToolRuntime(workspace, mode="auto"),
                                  store=store, on_event=on_event)
            resumed.resume(first.session_id)
            checks["resume_preserves_counters"] = (resumed.budget["spent"]["tool_calls"] == before["spent"]["tool_calls"] == 3
                                                   and resumed.budget["spent"]["side_effects"] == 1
                                                   and resumed.budget["spent"]["model_requests"] == 2)

            workspace = root / "output"
            workspace.mkdir()
            (workspace / "value.py").write_text("Я" * 100, encoding="utf-8")
            output = CodingAgent(BudgetDemoClient({"tool_calls": [read]}, {"content": "Done"}),
                                 ToolRuntime(workspace, mode="auto"), store=store,
                                 budget_limits=BudgetLimits(result_bytes=60), on_event=on_event)
            output.run("Read with a small output quota")
            result = store.calls(output.session_id)[0]["result"]["content"]
            checks["output_truncated_and_bounded"] = "TRUNCATED" in result and len(result.encode("utf-8")) <= 60

            workspace = root / "crash"
            workspace.mkdir()
            class CrashRuntime(ToolRuntime):
                def execute(self, name, arguments):
                    super().execute(name, arguments)
                    raise KeyboardInterrupt  # Between the effect and its result checkpoint.
            crashed = CodingAgent(BudgetDemoClient({"tool_calls": [write]}),
                                  CrashRuntime(workspace, mode="auto"), store=store, on_event=on_event)
            try:
                crashed.run("Write then interrupt")
            except KeyboardInterrupt:
                pass
            charged = crashed.budget
            try:
                CodingAgent(BudgetDemoClient(), ToolRuntime(workspace, mode="auto"), store=store).resume(crashed.session_id)
                checks["uncertain_call_blocked"] = False
            except UncertainActionError:
                checks["uncertain_call_blocked"] = True
            checks["crash_does_not_refund"] = crashed.budget == charged and charged["spent"]["result_bytes"] == 1024
            store.resolve(crashed.session_id, "1:0", "applied", lambda item: (workspace / "value.py").read_bytes() == b"fixed")
            checks["resolution_does_not_refund"] = crashed.budget == charged
    return {"checks": checks, "budgets": examples, "success": all(checks.values())}
