from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.cli import load_policy
from hw_ai_agent.policy import Policy
from hw_ai_agent.session import SessionError, SessionStore, UncertainActionError
from hw_ai_agent.tools import ToolError, ToolRuntime
from hw_ai_agent.workflow_demo import FIXED_TAX, call, run_workflow_demo


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = 0

    def complete(self, messages, tools):
        self.requests += 1
        return next(self.responses)


def write(name, content="fixed"):
    return call(name, "write_file", {"path": name + ".py", "content": content})


class ModeSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.store = SessionStore(self.root / ".agent/sessions.sqlite3")

    def tearDown(self):
        self.store.db.close()
        self.temporary.cleanup()

    def runtime(self, **kwargs):
        runtime = ToolRuntime(self.workspace, **kwargs)
        runtime.bash = Mock(return_value="exit_code=0\nOK")
        return runtime

    def test_safe_defaults_require_write_and_bash_permission(self):
        policy = load_policy(None, self.workspace)
        self.assertEqual(policy.decide("write_file").action.value, "ask")
        self.assertEqual(policy.decide("Bash").action.value, "ask")
        self.assertEqual(self.runtime().policy.decide("Bash").action.value, "ask")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.runtime(mode="unknown")

    def test_safe_asks_once_per_tool_for_multiple_files_and_commands(self):
        approve = Mock(return_value=True)
        runtime = self.runtime(approve=approve)
        client = Client({"tool_calls": [write("a"), write("b"),
                        call("ls", "Bash", {"command": "ls"}),
                        call("pwd", "bash", {"command": "pwd"})]}, {"content": "done"})
        CodingAgent(client, runtime, store=self.store).run("Task")
        self.assertEqual([c.args[0].tool for c in approve.call_args_list], ["write_file", "Bash"])
        self.assertTrue(all(c.args[0].scope == "session" for c in approve.call_args_list))
        self.assertEqual(runtime.bash.call_count, 2)
        self.assertTrue((self.workspace / "b.py").exists())

    def test_new_session_does_not_inherit_grants(self):
        approve = Mock(return_value=True)
        runtime = self.runtime(approve=approve)
        for name in ("a", "b"):
            client = Client({"tool_calls": [write(name), call(name + "cmd", "Bash", {"command": "pwd"})]},
                            {"content": "done"})
            CodingAgent(client, runtime, store=self.store).run("Task")
        self.assertEqual(approve.call_count, 4)

    def test_refusal_does_not_grant_session_permission(self):
        approve = Mock(return_value=False)
        runtime = self.runtime(approve=approve)
        agent = CodingAgent(Client({"tool_calls": [write("a"), write("b")]}, {"content": "denied"}),
                            runtime, store=self.store)
        agent.run("Task")
        self.assertEqual(approve.call_count, 2)
        self.assertEqual(runtime.session_permissions, set())
        self.assertFalse((self.workspace / "a.py").exists())

    def test_auto_runs_without_confirmation(self):
        approve = Mock(side_effect=AssertionError("No prompt expected"))
        runtime = self.runtime(mode="auto", approve=approve)
        CodingAgent(Client({"tool_calls": [write("a"), call("cmd", "Bash", {"command": "pwd"})]},
                           {"content": "done"}), runtime, store=self.store).run("Task")
        approve.assert_not_called()
        self.assertTrue((self.workspace / "a.py").exists())
        runtime.bash.assert_called_once()

    def test_auto_and_grants_do_not_override_policy_deny(self):
        for mode in ("safe", "auto"):
            runtime = self.runtime(mode=mode, policy=Policy.from_dict({"defaults": {"write_file": "deny", "Bash": "deny"}}))
            runtime.session_permissions = {"write_file", "Bash"}
            for tool, arguments in (("write_file", {"path": "a.py", "content": "x"}), ("Bash", {"command": "pwd"})):
                with self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
                    runtime.execute(tool, arguments)
            runtime.bash.assert_not_called()

    def test_auto_preserves_runtime_boundaries(self):
        runtime = self.runtime(mode="auto")
        for tool, arguments in (("write_file", {"path": "test_x.py", "content": "x"}),
                                ("write_file", {"path": "../outside.py", "content": "x"}),
                                ("Bash", {"command": "python -c 'print(1)'"})):
            with self.assertRaises(ToolError):
                runtime.execute(tool, arguments)

    def test_auto_does_not_grant_ask_read_access(self):
        (self.workspace / "private.py").write_text("private")
        runtime = self.runtime(mode="auto", policy=Policy.from_dict({"defaults": {"read_file": "ask"}}))
        with self.assertRaisesRegex(ToolError, "APPROVAL_REQUIRED"):
            runtime.execute("read_file", {"path": "private.py"})

    def test_resume_restores_permission_without_prompt(self):
        def stop(event):
            if event["event"] == "tool_result":
                raise KeyboardInterrupt
        first = CodingAgent(Client({"tool_calls": [write("a"), write("b")]}),
                            self.runtime(approve=lambda request: True), store=self.store, on_event=stop)
        with self.assertRaises(KeyboardInterrupt):
            first.run("Task")
        approve = Mock(return_value=False)
        CodingAgent(Client({"content": "done"}), self.runtime(approve=approve), store=self.store).resume(first.session_id)
        approve.assert_not_called()
        self.assertEqual((self.workspace / "b.py").read_text(), "fixed")

    def test_resume_refuses_execution_mode_change(self):
        first = CodingAgent(Client({"content": "done"}), self.runtime(mode="safe"), store=self.store)
        first.run("Task")
        with self.assertRaises(SessionError):
            CodingAgent(Client(), self.runtime(mode="auto"), store=self.store).resume(first.session_id)

    def test_legacy_single_call_permission_is_not_promoted_to_session_grant(self):
        runtime = self.runtime(approve=lambda request: False)
        agent = CodingAgent(Client({"content": "done"}), runtime, store=self.store)
        session_id = self.store.create(agent._binding(), [{"role": "user", "content": "Task"}], 12, 120000)
        self.store.event(session_id, {"event": "policy_decision", "tool": "write_file", "outcome": "approved"})
        self.store.assistant(session_id, {"role": "assistant", "tool_calls": [write("a")]}, 0)
        agent.resume(session_id)
        self.assertFalse((self.workspace / "a.py").exists())

    def uncertain(self, tool="write_file"):
        runtime = self.runtime(mode="auto")
        original = runtime.execute
        def execute_then_interrupt(name, arguments):
            original(name, arguments)
            raise KeyboardInterrupt
        runtime.execute = execute_then_interrupt
        action = write("a") if tool == "write_file" else call("pwd", "Bash", {"command": "pwd"})
        agent = CodingAgent(Client({"tool_calls": [action]}), runtime, store=self.store)
        with self.assertRaises(KeyboardInterrupt):
            agent.run("Task")
        return agent.session_id

    def test_uncertain_bash_does_not_retry_silently(self):
        session_id = self.uncertain("Bash")
        runtime = self.runtime(mode="auto")
        client = Client()
        with self.assertRaises(UncertainActionError):
            CodingAgent(client, runtime, store=self.store).resume(session_id)
        runtime.bash.assert_not_called()
        self.assertEqual(client.requests, 0)
        self.assertEqual(self.store.calls(session_id, True)[0]["state"], "uncertain")

    def test_abort_is_terminal_and_executes_nothing(self):
        session_id = self.uncertain()
        self.store.resolve(session_id, "1:0", "abort")
        self.assertEqual(self.store.get(session_id).status, "aborted")
        client = Client()
        runtime = self.runtime(mode="auto")
        with self.assertRaisesRegex(SessionError, "terminated"):
            CodingAgent(client, runtime, store=self.store).resume(session_id)
        with self.assertRaises(SessionError):
            self.store.resolve(session_id, "1:0", "retry")
        self.assertEqual(client.requests, 0)
        self.assertEqual((self.workspace / "a.py").read_text(), "fixed")

    def test_retry_of_unknown_instrument_is_rejected(self):
        session_id = self.uncertain()
        action = call("dangerous", "append_external", {"content": "x"})
        self.store.db.execute("UPDATE calls SET call=? WHERE session_id=?", (json.dumps(action), session_id))
        with self.assertRaisesRegex(SessionError, "retry contract"):
            self.store.resolve(session_id, "1:0", "retry")


@unittest.skipUnless(os.getenv("REQUIRE_SANDBOX_TESTS") == "1", "Requires native macOS sandbox")
class WorkflowIntegrationTests(unittest.TestCase):
    def test_safe_multifile_demo_asks_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approve = Mock(return_value=True)
            report = run_workflow_demo(root / "workspace", root / ".agent/state.sqlite3", approve=approve)
            self.assertTrue(report["success"])
            self.assertEqual(report["permission_requests"], {"write_file": 1, "Bash": 1})
            self.assertEqual(approve.call_count, 2)
            self.assertEqual(report["write_attempts"], {"tax.py": 1, "receipt.py": 1})

    def test_auto_multifile_demo_does_not_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_workflow_demo(root / "workspace", root / ".agent/state.sqlite3", mode="auto")
            self.assertTrue(report["success"])
            self.assertEqual(report["permission_requests"], {})

    def test_real_process_crash_and_all_three_resolutions(self):
        for resolution in ("retry", "skip", "abort"):
            with self.subTest(resolution=resolution), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workspace, db = root / "workspace", root / ".agent/state.sqlite3"
                env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
                def cli(*args):
                    return subprocess.run([sys.executable, "-m", "hw_ai_agent.cli", *args],
                                          cwd=root, env=env, capture_output=True, text=True, timeout=20)
                crashed = cli("workflow-demo", "--mode", "auto", "--workspace", str(workspace),
                              "--state-db", str(db), "--crash-after-write")
                self.assertEqual(crashed.returncode, 86, crashed.stderr)
                self.assertEqual((workspace / "tax.py").read_text(), FIXED_TAX)
                with SessionStore(db) as store:
                    session_id = store.list_sessions()[0]["id"]
                    self.assertEqual(store.calls(session_id, True)[0]["state"], "running")
                blocked = cli("resume", session_id, "--state-db", str(db))
                self.assertEqual(blocked.returncode, 1, blocked.stdout + blocked.stderr)
                self.assertIn("unknown result", blocked.stderr)
                resolved = cli("resolve", session_id, "--state-db", str(db), "--call", "1:0", "--resolution", resolution)
                self.assertEqual(resolved.returncode, 1 if resolution == "abort" else 0, resolved.stderr)
                resumed = cli("resume", session_id, "--state-db", str(db))
                self.assertEqual(resumed.returncode, 1 if resolution == "abort" else 0, resumed.stdout + resumed.stderr)
                with SessionStore(db) as store:
                    self.assertEqual(store.get(session_id).status, "aborted" if resolution == "abort" else "completed")
                    first_call = store.calls(session_id)[0]
                    if resolution == "skip":
                        self.assertEqual(first_call["state"], "failed")
                        self.assertIn("RECOVERY_SKIPPED", first_call["result"]["content"])
                if resolution != "abort":
                    report = json.loads((workspace / "workflow-report.json").read_text())
                    self.assertTrue(report["success"])
                    self.assertEqual(report["write_attempts"]["tax.py"], 2 if resolution == "retry" else 1)
                    self.assertNotIn(str(workspace), json.dumps(report))


if __name__ == "__main__":
    unittest.main()
