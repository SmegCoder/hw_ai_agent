from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
from pathlib import Path

from hw_ai_agent.agent import AgentError, CodingAgent
from hw_ai_agent.config import Settings
from hw_ai_agent.policy import Policy
from hw_ai_agent.session import SessionBusyError, SessionError, SessionStore, UncertainActionError
from hw_ai_agent.tools import ToolRuntime


def call(call_id, name, arguments):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(list(messages))
        item = next(self.responses)
        if isinstance(item, BaseException):
            raise item
        return item


class CountingRuntime(ToolRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executions = []

    def execute(self, name, arguments):
        self.executions.append(name)
        return super().execute(name, arguments)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.path = self.root / ".agent/sessions.sqlite3"
        self.store = SessionStore(self.path)
        self.policy = Policy.from_dict({"defaults": {"write_file": "allow"}})

    def tearDown(self):
        self.store.db.close()
        self.temporary.cleanup()

    def runtime(self, **kwargs):
        return CountingRuntime(self.workspace, policy=kwargs.pop("policy", self.policy), mode=kwargs.pop("mode", "auto"),
                               approve=kwargs.pop("approve", lambda request: True), **kwargs)

    def interrupted_after_write(self):
        def interrupt(event):
            if event["event"] == "tool_result" and event["name"] == "write_file":
                raise KeyboardInterrupt
        first = CodingAgent(Client({"tool_calls": [
            call("write", "write_file", {"path": "order.py", "content": "saved"}),
            call("read", "read_file", {"path": "order.py"})]}), self.runtime(), store=self.store, on_event=interrupt)
        with self.assertRaises(KeyboardInterrupt):
            first.run("Task")
        return first.session_id

    def uncertain_write(self, policy=None, approve=None):
        runtime = self.runtime(policy=policy or self.policy, approve=approve or (lambda request: True), mode="safe" if approve else "auto")
        def execute_then_die(name, arguments):
            ToolRuntime.execute(runtime, name, arguments)
            raise KeyboardInterrupt
        runtime.execute = execute_then_die
        agent = CodingAgent(Client({"tool_calls": [call("write", "write_file", {"path": "order.py", "content": "effect"})]}),
                            runtime, store=self.store)
        with self.assertRaises(KeyboardInterrupt):
            agent.run("Write file")
        return agent.session_id

    def test_saved_result_is_not_reexecuted_after_reopening_database(self):
        session_id = self.interrupted_after_write()
        self.store.db.close()
        self.store = SessionStore(self.path)
        runtime = self.runtime()
        client = Client({"content": "Finished"})
        agent = CodingAgent(client, runtime, store=self.store)
        self.assertEqual(agent.resume(session_id), "Finished")
        self.assertEqual(runtime.executions, ["read_file"])
        self.assertEqual((self.workspace / "order.py").read_text(), "saved")
        self.assertEqual([m["tool_call_id"] for m in client.requests[0] if m["role"] == "tool"], ["write", "read"])
        self.assertEqual(self.store.get(session_id).status, "completed")

    def test_interruption_before_execution_keeps_pending_write(self):
        # Pause immediately after the assistant and its call list are checkpointed.
        original = self.store.assistant
        def checkpoint_then_interrupt(*args):
            original(*args)
            raise KeyboardInterrupt
        self.store.assistant = checkpoint_then_interrupt
        first = CodingAgent(Client({"tool_calls": [call("write", "write_file", {"path": "new.py", "content": "x"})]}),
                            self.runtime(), store=self.store)
        with self.assertRaises(KeyboardInterrupt):
            first.run("Task")
        self.assertFalse((self.workspace / "new.py").exists())
        self.store.assistant = original
        second = CodingAgent(Client({"content": "Done"}), self.runtime(), store=self.store)
        self.assertEqual(second.resume(first.session_id), "Done")
        self.assertEqual((self.workspace / "new.py").read_text(), "x")

    def test_uncertain_write_is_blocked_without_model_or_tool_retry(self):
        session_id = self.uncertain_write()
        runtime = self.runtime()
        client = Client()
        with self.assertRaises(UncertainActionError):
            CodingAgent(client, runtime, store=self.store).resume(session_id)
        self.assertEqual(runtime.executions, [])
        self.assertEqual(client.requests, [])
        self.assertEqual(self.store.get(session_id).status, "blocked")
        self.assertEqual((self.workspace / "order.py").read_text(), "effect")

    def test_applied_resolution_avoids_repeat_write(self):
        session_id = self.uncertain_write()
        self.store.resolve(session_id, "1:0", "applied", lambda item:
                           (self.workspace / "order.py").read_text() == json.loads(item["call"]["function"]["arguments"])["content"])
        runtime = self.runtime()
        self.assertEqual(CodingAgent(Client({"content": "Done"}), runtime, store=self.store).resume(session_id), "Done")
        self.assertEqual(runtime.executions, [])

    def test_applied_resolution_requires_actual_verification(self):
        session_id = self.uncertain_write()
        with self.assertRaises(SessionError):
            self.store.resolve(session_id, "1:0", "applied", lambda item: False)
        self.assertEqual(self.store.calls(session_id, True)[0]["state"], "running")

    def test_explicit_retry_requires_new_approval_in_safe(self):
        session_id = self.uncertain_write(policy=Policy(), approve=lambda request: True)
        self.store.resolve(session_id, "1:0", "retry")
        requested = []
        runtime = self.runtime(policy=Policy(), mode="safe", approve=lambda request: requested.append(request) or True)
        self.assertEqual(CodingAgent(Client({"content": "Done"}), runtime, store=self.store).resume(session_id), "Done")
        self.assertEqual(len(requested), 1)
        self.assertEqual(runtime.executions, ["write_file"])

    def test_skip_resolution_returns_error_to_model(self):
        session_id = self.uncertain_write()
        self.store.resolve(session_id, "1:0", "skip")
        client = Client({"content": "Skipped"})
        runtime = self.runtime()
        CodingAgent(client, runtime, store=self.store).resume(session_id)
        self.assertIn("RECOVERY_SKIPPED", client.requests[0][-1]["content"])
        self.assertEqual(runtime.executions, [])

    def test_readonly_interruption_can_be_retried(self):
        (self.workspace / "order.py").write_text("x")
        first_runtime = self.runtime()
        first_runtime.execute = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
        agent = CodingAgent(Client({"tool_calls": [call("read", "read_file", {"path": "order.py"})]}),
                            first_runtime, store=self.store)
        with self.assertRaises(KeyboardInterrupt):
            agent.run("Read file")
        resumed_runtime = self.runtime()
        with self.assertRaises(UncertainActionError):
            CodingAgent(Client(), resumed_runtime, store=self.store).resume(agent.session_id)
        self.store.resolve(agent.session_id, "1:0", "retry")
        CodingAgent(Client({"content": "Done"}), resumed_runtime, store=self.store).resume(agent.session_id)
        self.assertEqual(resumed_runtime.executions, ["read_file"])

    def test_model_failure_preserves_round_counter(self):
        first = CodingAgent(Client(RuntimeError("offline")), self.runtime(), store=self.store, max_rounds=1)
        with self.assertRaises(RuntimeError):
            first.run("Task")
        self.assertEqual(self.store.get(first.session_id).rounds, 1)
        resumed = Client({"content": "Unexpected"})
        with self.assertRaises(AgentError):
            CodingAgent(resumed, self.runtime(), store=self.store, max_rounds=99).resume(first.session_id)
        self.assertEqual(resumed.requests, [])

    def test_completed_session_uses_cached_answer(self):
        first = CodingAgent(Client({"content": "Saved answer"}), self.runtime(), store=self.store)
        first.run("Task")
        resumed = Client()
        self.assertEqual(CodingAgent(resumed, self.runtime(), store=self.store).resume(first.session_id), "Saved answer")
        self.assertEqual(resumed.requests, [])

    def test_final_assistant_checkpoint_survives_crash_before_complete(self):
        original = self.store.complete
        self.store.complete = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
        first = CodingAgent(Client({"content": "Final"}), self.runtime(), store=self.store)
        with self.assertRaises(KeyboardInterrupt):
            first.run("Task")
        self.store.complete = original
        self.assertEqual(CodingAgent(Client(), self.runtime(), store=self.store).resume(first.session_id), "Final")

    def test_policy_and_workspace_changes_refuse_continuation(self):
        session_id = self.interrupted_after_write()
        changed = self.runtime(policy=Policy())
        with self.assertRaisesRegex(SessionError, "policy"):
            CodingAgent(Client(), changed, store=self.store).resume(session_id)
        another = self.root / "other"
        another.mkdir()
        with self.assertRaisesRegex(SessionError, "workspace"):
            CodingAgent(Client(), CountingRuntime(another, policy=self.policy), store=self.store).resume(session_id)

    def test_changed_backend_is_rejected_and_key_is_not_saved(self):
        client = Client({"content": "Done"})
        client.settings = Settings("openrouter", "model-a", "https://example.com/v1", "secret-not-for-storage")
        first = CodingAgent(client, self.runtime(), store=self.store)
        first.run("Task")
        self.assertNotIn(b"secret-not-for-storage", self.path.read_bytes())
        other = Client()
        other.settings = Settings("openrouter", "model-b", "https://example.com/v1", "new-secret")
        with self.assertRaisesRegex(SessionError, "model"):
            CodingAgent(other, self.runtime(), store=self.store).resume(first.session_id)

    def test_session_cannot_be_run_or_resolved_twice_concurrently(self):
        session_id = self.interrupted_after_write()
        with self.store.lock(session_id):
            with self.assertRaises(SessionBusyError):
                CodingAgent(Client(), self.runtime(), store=self.store).resume(session_id)
            with self.assertRaises(SessionBusyError):
                self.store.resolve(session_id, "1:0", "skip")

    def test_database_inside_workspace_is_rejected(self):
        with SessionStore(self.workspace / "bad.db") as store:
            with self.assertRaisesRegex(SessionError, "outside"):
                CodingAgent(Client(), self.runtime(), store=store).run("Task")

    def test_invalid_id_and_corrupted_history_fail_closed(self):
        with self.assertRaises(SessionError):
            self.store.get("../../escape")
        first = CodingAgent(Client({"content": "Done"}), self.runtime(), store=self.store)
        first.run("Task")
        self.store.db.execute("UPDATE sessions SET messages='not json' WHERE id=?", (first.session_id,))
        with self.assertRaises(SessionError):
            self.store.get(first.session_id)

    def test_database_permissions_are_private(self):
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)

    def test_hard_process_exit_after_effect_is_not_repeated(self):
        script = '''import os,sys,json
from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.tools import ToolRuntime
from hw_ai_agent.policy import Policy
from hw_ai_agent.session import SessionStore
class Runtime(ToolRuntime):
    def execute(self,name,args):
        result=super().execute(name,args)
        os._exit(23)
class Client:
    def complete(self,messages,tools):
        return {"tool_calls":[{"id":"write","function":{"name":"write_file","arguments":json.dumps({"path":"order.py","content":"effect"})}}]}
with SessionStore(sys.argv[2]) as store:
    CodingAgent(Client(),Runtime(sys.argv[1],policy=Policy.from_dict({"defaults":{"write_file":"allow"}}),mode="auto",approve=lambda request: True),store=store).run("Task")
'''
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run([sys.executable, "-c", script, str(self.workspace), str(self.path)], env=env, timeout=10, capture_output=True)
        self.assertEqual(result.returncode, 23, result.stderr.decode())
        session_id = self.store.list_sessions()[0]["id"]
        with self.assertRaises(UncertainActionError):
            CodingAgent(Client(), self.runtime(), store=self.store).resume(session_id)
        self.assertEqual((self.workspace / "order.py").read_text(), "effect")

    def test_result_checkpoint_failure_blocks_automatic_retry(self):
        original = self.store.finish_call
        self.store.finish_call = lambda *args: (_ for _ in ()).throw(SessionError("checkpoint failed"))
        first = CodingAgent(Client({"tool_calls": [call("write", "write_file", {"path": "order.py", "content": "effect"})]}),
                            self.runtime(), store=self.store)
        with self.assertRaises(SessionError):
            first.run("Task")
        self.store.finish_call = original
        runtime = self.runtime()
        with self.assertRaises(UncertainActionError):
            CodingAgent(Client(), runtime, store=self.store).resume(first.session_id)
        self.assertEqual(runtime.executions, [])
        self.assertEqual((self.workspace / "order.py").read_text(), "effect")

    def test_credentials_cannot_be_added_to_custom_metadata(self):
        agent = CodingAgent(Client(), self.runtime(), store=self.store, session_metadata={"api_key": "private"})
        with self.assertRaises(SessionError):
            agent.run("Task")
        self.assertEqual(self.store.list_sessions(), [])


class SessionCLITests(unittest.TestCase):
    def test_applied_verification_compares_raw_bytes_including_crlf(self):
        from hw_ai_agent.cli import verify_applied
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = ToolRuntime(root)
            content = "first\r\nsecond\r\n"
            (root / "order.py").write_bytes(content.encode())
            item = {"call": call("write", "write_file", {"path": "order.py", "content": content})}
            self.assertTrue(verify_applied(runtime, item))
            (root / "order.py").write_bytes(content.replace("\r", "").encode())
            self.assertFalse(verify_applied(runtime, item))

    def test_interactive_confirmation_still_works(self):
        from hw_ai_agent.cli import confirm_action
        from hw_ai_agent.policy import ApprovalRequest
        request = ApprovalRequest("write_file", "order.py", "ask", "diff", "fingerprint")
        with patch("hw_ai_agent.cli.sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="y"), redirect_stdout(io.StringIO()):
            self.assertTrue(confirm_action(request))

    def test_stop_listener_only_interrupts_after_tool_result(self):
        from hw_ai_agent.cli import event_listener
        with redirect_stdout(io.StringIO()):
            listener = event_listener(1)
            listener({"event": "model_request", "round": 1})
            listener({"event": "policy_decision", "tool": "read_file", "action": "allow", "outcome": "permitted"})
            with self.assertRaises(KeyboardInterrupt):
                listener({"event": "tool_result", "name": "read_file", "content": "saved"})


@unittest.skipUnless(os.getenv("REQUIRE_SANDBOX_TESTS") == "1", "Requires native macOS sandbox")
class SessionDemoTests(unittest.TestCase):
    def test_interrupted_demo_is_resumed_without_repeating_write(self):
        from hw_ai_agent.session_demo import run_session_demo
        report = run_session_demo()
        self.assertTrue(report["success"])

    def test_cli_pause_and_resume_use_the_same_session(self):
        from hw_ai_agent.cli import main
        from hw_ai_agent.demo import initialize_demo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            initialize_demo(workspace)
            policy_file = root / "policy.json"
            policy_file.write_text(json.dumps({
                "directories": {"allow": ["."], "deny": []},
                "defaults": {"read_file": "allow", "write_file": "allow", "Bash": "allow"},
            }))
            db = root / ".agent/sessions.sqlite3"
            code = ("def calculate_total(prices, discount_percent):\n"
                    "    if not 0 <= discount_percent <= 100:\n"
                    "        raise ValueError('invalid')\n"
                    "    return sum(prices) * (1 - discount_percent / 100)\n")
            def fake_client(settings, **kwargs):
                c = Client({"tool_calls": [
                    call("read", "read_file", {"path": "order.py"}),
                    call("write", "write_file", {"path": "order.py", "content": code}),
                    call("tests", "Bash", {"command": "python -m unittest discover -v"})]})
                c.settings, c.max_tokens, c.usage = settings, kwargs["max_tokens"], []
                return c
            first_argv = ["hw-ai-agent", "run", "--provider", "ollama", "--model", "test-model",
                "--workspace", str(workspace), "--state-db", str(db), "--policy", str(policy_file),
                "--mode", "auto", "--stop-after-tools", "1", "Fix order"]
            with patch("sys.argv", first_argv), patch("hw_ai_agent.cli.OpenAICompatibleClient", side_effect=fake_client), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 130)
            with SessionStore(db) as store:
                session_id = store.list_sessions()[0]["id"]
                self.assertEqual(store.get(session_id).status, "interrupted")
            def final_client(settings, **kwargs):
                c = Client({"content": "All tests pass."})
                c.settings, c.max_tokens, c.usage = settings, kwargs["max_tokens"], []
                return c
            with patch("sys.argv", ["hw-ai-agent", "resume", session_id, "--state-db", str(db)]), \
                 patch("hw_ai_agent.cli.confirm_action", return_value=True), \
                 patch("hw_ai_agent.cli.OpenAICompatibleClient", side_effect=final_client), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            with SessionStore(db) as store:
                self.assertEqual(store.get(session_id).status, "completed")
                writes = [c for c in store.calls(session_id) if c["call"]["function"]["name"] == "write_file"]
                self.assertEqual(len(writes), 1)
                self.assertEqual(writes[0]["state"], "completed")


if __name__ == "__main__":
    unittest.main()
