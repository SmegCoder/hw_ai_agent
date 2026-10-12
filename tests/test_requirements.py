from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from hw_ai_agent.agent import AgentError, CodingAgent
from hw_ai_agent.policy import ApprovalDeferred, Policy, PolicyError
from hw_ai_agent.session import SessionStore
from hw_ai_agent.tools import ToolError, ToolRuntime
from hw_ai_agent.workflow_demo import call


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(list(messages))
        return next(self.responses)


def write(identifier, content="fixed"):
    return call(identifier, "write_file", {"path": identifier + ".py", "content": content})


class WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.store = SessionStore(self.root / ".agent/state.sqlite3")

    def tearDown(self):
        self.store.db.close()
        self.temporary.cleanup()

    def runtime(self, **kwargs):
        runtime = ToolRuntime(self.workspace, **kwargs)
        runtime.bash = Mock(return_value="exit_code=0\nOK")
        return runtime


class ApprovalContractTests(WorkspaceCase):
    def test_reads_never_need_mode_approval(self):
        (self.workspace / "a.py").write_text("original")
        for mode in ("safe", "auto"):
            approve = Mock(side_effect=AssertionError("Reading must not prompt"))
            runtime = self.runtime(mode=mode, approve=approve)
            self.assertEqual(runtime.execute("read_file", {"path": "a.py"}), "original")
            self.assertIn("a.py", runtime.execute("list_files", {}))
            approve.assert_not_called()

    def test_auto_allow_runs_without_first_approval(self):
        approve = Mock(return_value=False)
        policy = Policy.from_dict({"defaults": {"write_file": "allow", "Bash": "allow"}})
        runtime = self.runtime(mode="auto", policy=policy, approve=approve)
        CodingAgent(Client({"tool_calls": [write("a"), call("cmd", "Bash", {"command": "pwd"}), write("b")]},
                           {"content": "done"}), runtime, store=self.store).run("Task")
        approve.assert_not_called()
        runtime.bash.assert_called_once()
        self.assertTrue((self.workspace / "a.py").exists())
        self.assertTrue((self.workspace / "b.py").exists())
        self.assertEqual(runtime.session_permissions, set())

    def test_auto_ask_write_policy_does_not_prompt(self):
        approve = Mock(return_value=False)
        runtime = self.runtime(mode="auto", approve=approve)
        CodingAgent(Client({"tool_calls": [write("a"), write("b"), write("c")]}, {"content": "done"}),
                    runtime, store=self.store).run("Task")
        approve.assert_not_called()
        self.assertTrue(all((self.workspace / (name + ".py")).exists() for name in ("a", "b", "c")))

    def test_new_auto_sessions_never_require_write_or_bash_approval(self):
        approve = Mock(return_value=False)
        runtime = self.runtime(mode="auto", approve=approve)
        for name in ("a", "b"):
            CodingAgent(Client({"tool_calls": [write(name), call("cmd", "Bash", {"command": "pwd"})]},
                               {"content": "done"}), runtime, store=self.store).run("Task")
        approve.assert_not_called()
        self.assertEqual(runtime.bash.call_count, 2)

    def test_auto_resume_needs_no_grants_after_database_reopen(self):
        completed = 0
        def stop(event):
            nonlocal completed
            if event["event"] == "tool_result":
                completed += 1
                if completed == 2:
                    raise KeyboardInterrupt
        actions = [write("a"), call("pwd", "Bash", {"command": "pwd"}),
                   write("b"), call("ls", "Bash", {"command": "ls"})]
        first = CodingAgent(Client({"tool_calls": actions}), self.runtime(mode="auto", approve=lambda request: True),
                            store=self.store, on_event=stop)
        with self.assertRaises(KeyboardInterrupt):
            first.run("Task")
        self.store.db.close()
        self.store = SessionStore(self.root / ".agent/state.sqlite3")
        approve = Mock(return_value=False)
        runtime = self.runtime(mode="auto", approve=approve)
        CodingAgent(Client({"content": "done"}), runtime, store=self.store).resume(first.session_id)
        approve.assert_not_called()
        runtime.bash.assert_called_once_with(command="ls")
        self.assertTrue((self.workspace / "b.py").exists())
        self.assertEqual(runtime.session_permissions, set())

    def test_standalone_auto_executes_writes_without_a_session(self):
        approve = Mock(return_value=False)
        runtime = self.runtime(mode="auto", approve=approve)
        runtime.execute("write_file", {"path": "a.py", "content": "one"})
        runtime.execute("write_file", {"path": "b.py", "content": "two"})
        approve.assert_not_called()
        self.assertEqual((self.workspace / "b.py").read_text(), "two")

    def test_non_boolean_or_broken_auto_deletion_approval_never_grants_permission(self):
        for answer in (1, "yes", None, RuntimeError("broken")):
            approve = Mock(side_effect=answer) if isinstance(answer, Exception) else Mock(return_value=answer)
            runtime = self.runtime(mode="auto", approve=approve)
            CodingAgent(Client({"tool_calls": [call("a", "delete_file", {"path": "a.py"}),
                                                  call("b", "delete_file", {"path": "b.py"})]}, {"content": "denied"}),
                        runtime, store=self.store).run("Task")
            self.assertEqual(approve.call_count, 2)
            self.assertEqual(runtime.session_permissions, set())
            self.assertFalse((self.workspace / "a.py").exists())

    def test_cached_delete_grant_never_authorizes_deletion(self):
        (self.workspace / "a.py").write_text("original")
        approve = Mock(return_value=False)
        runtime = self.runtime(mode="auto", approve=approve)
        runtime.session_permissions = {"write_file", "Bash", "delete_file"}
        with self.assertRaises(ToolError):
            runtime.execute("delete_file", {"path": "a.py"})
        approve.assert_called_once()
        self.assertEqual(approve.call_args.args[0].scope, "call")
        self.assertTrue((self.workspace / "a.py").exists())

    def test_legacy_grants_do_not_authorize_deletion_after_resume(self):
        (self.workspace / "a.py").write_text("original")
        runtime = self.runtime(mode="auto")
        agent = CodingAgent(Client(), runtime, store=self.store)
        metadata = agent._binding()
        metadata.pop("approval_contract")
        metadata["policy"].pop("directories")
        identifier = self.store.create(metadata, [{"role": "user", "content": "Task"}], 12, 120000)
        self.store.event(identifier, {"event": "policy_decision", "tool": "delete_file", "outcome": "approved", "scope": "session"})
        self.store.assistant(identifier, {"role": "assistant", "tool_calls": [call("remove", "delete_file", {"path": "a.py"})]}, 0)
        approve = Mock(return_value=False)
        CodingAgent(Client({"content": "done"}), self.runtime(mode="auto", approve=approve), store=self.store).resume(identifier)
        approve.assert_called_once()
        self.assertEqual(self.store.get(identifier).metadata["approval_contract"], 2)
        self.assertTrue((self.workspace / "a.py").exists())

    def test_deferred_confirmation_is_saved_and_can_be_declined(self):
        for mode in ("safe", "auto"):
            with self.subTest(mode=mode):
                runtime = self.runtime(mode=mode, approve=Mock(side_effect=ApprovalDeferred("waiting")))
                first = CodingAgent(Client({"tool_calls": [call("remove", "delete_file", {"path": "a.py"})]}), runtime, store=self.store)
                with self.assertRaises(ApprovalDeferred):
                    first.run("Task")
                self.assertEqual(self.store.get(first.session_id).status, "waiting_approval")
                self.assertEqual(self.store.calls(first.session_id, True)[0]["state"], "awaiting_approval")
                self.assertFalse((self.workspace / "a.py").exists())
                client = Client({"content": "declined"})
                CodingAgent(client, self.runtime(mode=mode, approve=lambda request: False), store=self.store).resume(first.session_id)
                self.assertIn("APPROVAL_REQUIRED", client.requests[0][-1]["content"])
                self.assertFalse((self.workspace / "a.py").exists())

    def test_interrupt_during_approval_is_not_an_uncertain_effect(self):
        first = CodingAgent(Client({"tool_calls": [write("a")]}),
                            self.runtime(approve=Mock(side_effect=KeyboardInterrupt)), store=self.store)
        with self.assertRaises(KeyboardInterrupt):
            first.run("Task")
        self.assertEqual(self.store.calls(first.session_id, True)[0]["state"], "awaiting_approval")
        CodingAgent(Client({"content": "done"}), self.runtime(approve=lambda request: True), store=self.store).resume(first.session_id)
        self.assertTrue((self.workspace / "a.py").exists())

    def test_metadata_only_change_during_approval_does_not_block_write(self):
        path = self.workspace / "a.py"
        path.write_text("original")
        def approve(request):
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns + 1, info.st_mtime_ns))
            return True
        self.runtime(approve=approve).execute("write_file", {"path": "a.py", "content": "new"})
        self.assertEqual(path.read_text(), "new")

    def test_same_size_edit_with_restored_mtime_is_still_blocked(self):
        path = self.workspace / "a.py"
        path.write_text("original")
        def approve(request):
            info = path.stat()
            path.write_text("modified")
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
            return True
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            self.runtime(approve=approve).execute("write_file", {"path": "a.py", "content": "new"})
        self.assertEqual(path.read_text(), "modified")


class ProtocolReplayTests(WorkspaceCase):
    def test_completed_id_is_cached_without_second_write_or_prompt(self):
        for stored in (False, True):
            with self.subTest(stored=stored):
                action = write("a")
                approve = Mock(return_value=True)
                runtime = self.runtime(approve=approve)
                agent = CodingAgent(Client({"tool_calls": [action]}, {"tool_calls": [action]}, {"content": "done"}),
                                    runtime, store=self.store if stored else None)
                from hw_ai_agent.filesystem import write_target
                with patch("hw_ai_agent.tools.write_target", wraps=write_target) as writer:
                    agent.run("Task")
                writer.assert_called_once()
                approve.assert_called_once()
                self.assertEqual(len([m for m in agent.messages if m["role"] == "tool"]), 2)

    def test_changed_id_fails_before_any_call_in_the_batch(self):
        for stored in (False, True):
            with self.subTest(stored=stored):
                action = write("a")
                changed = write("a", "different")
                agent = CodingAgent(Client({"tool_calls": [action]}, {"tool_calls": [write("b"), changed]}),
                                    self.runtime(approve=lambda request: True), store=self.store if stored else None)
                with self.assertRaisesRegex(AgentError, "PROTOCOL_ERROR"):
                    agent.run("Task")
                self.assertEqual((self.workspace / "a.py").read_text(), "fixed")
                self.assertFalse((self.workspace / "b.py").exists())

    def test_replayed_id_after_resume_uses_saved_result(self):
        def stop(event):
            if event["event"] == "tool_result":
                raise KeyboardInterrupt
        first = CodingAgent(Client({"tool_calls": [write("a")]}), self.runtime(approve=lambda request: True),
                            store=self.store, on_event=stop)
        with self.assertRaises(KeyboardInterrupt):
            first.run("Task")
        approve = Mock(return_value=False)
        resumed = CodingAgent(Client({"tool_calls": [write("a")]}, {"content": "done"}),
                              self.runtime(approve=approve), store=self.store)
        resumed.resume(first.session_id)
        approve.assert_not_called()
        self.assertEqual((self.workspace / "a.py").read_text(), "fixed")

    def test_same_id_with_a_different_tool_is_a_protocol_error(self):
        action = write("a")
        changed = call("a", "read_file", {"path": "a.py"})
        agent = CodingAgent(Client({"tool_calls": [action]}, {"tool_calls": [changed]}),
                            self.runtime(approve=lambda request: True), store=self.store)
        with self.assertRaisesRegex(AgentError, "PROTOCOL_ERROR"):
            agent.run("Task")

    def test_retry_records_a_new_attempt_with_the_original_id(self):
        runtime = self.runtime(approve=lambda request: True)
        execute = runtime.execute
        def crash(name, arguments):
            execute(name, arguments)
            raise KeyboardInterrupt
        runtime.execute = crash
        first = CodingAgent(Client({"tool_calls": [write("a")]}), runtime, store=self.store)
        with self.assertRaises(KeyboardInterrupt):
            first.run("Task")
        self.store.resolve(first.session_id, "1:0", "retry")
        CodingAgent(Client({"content": "done"}), self.runtime(approve=lambda request: True), store=self.store).resume(first.session_id)
        attempts = [e for e in self.store.get(first.session_id).events if e.get("event") == "tool_call"]
        self.assertEqual([e["attempt"] for e in attempts], [1, 2])
        self.assertEqual([e["tool_call_id"] for e in attempts], ["a", "a"])


class DirectoryCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "allowed").mkdir()
        (self.root / "private").mkdir()
        (self.root / "private/data.txt").write_text("private")
        (self.root / "allowed/data.txt").write_text("public")

    def tearDown(self):
        self.temporary.cleanup()


class DirectoryPolicyTests(DirectoryCase):
    def test_directory_deny_applies_to_files_listing_and_bash_cwd(self):
        policy = Policy.from_dict({"directories": {"allow": ["."], "deny": ["private"]}})
        approve = Mock(return_value=True)
        runtime = ToolRuntime(self.root, policy=policy, approve=approve)
        for name, args in (("read_file", {"path": "private/data.txt"}),
                           ("write_file", {"path": "private/new.txt", "content": "x"}),
                           ("delete_file", {"path": "private/data.txt"}),
                           ("list_files", {"path": "private"}),
                           ("Bash", {"command": "pwd", "cwd": "private"})):
            with self.subTest(tool=name), self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
                runtime.execute(name, args)
        approve.assert_not_called()
        self.assertNotIn("private", runtime.execute("list_files", {"recursive": True}))
        self.assertEqual(runtime.execute("read_file", {"path": "allowed/data.txt"}), "public")

    def test_unlisted_directories_are_denied(self):
        policy = Policy.from_dict({"directories": {"allow": ["allowed"], "deny": []}})
        runtime = ToolRuntime(self.root, policy=policy)
        with self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
            runtime.execute("read_file", {"path": "private/data.txt"})

    def test_directory_symlink_cannot_bypass_a_deny(self):
        (self.root / "alias").symlink_to(self.root / "private", target_is_directory=True)
        policy = Policy.from_dict({"directories": {"allow": ["."], "deny": ["private"]}})
        with self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
            ToolRuntime(self.root, policy=policy).execute("read_file", {"path": "alias/data.txt"})

    def test_invalid_directory_rules_and_noncanonical_policy_fail_closed(self):
        for value in (["../outside"], ["/etc"], ["private/*"], "private", [1]):
            with self.subTest(value=value), self.assertRaises(PolicyError):
                Policy.from_dict({"directories": {"allow": value}})
        (self.root / "alias").symlink_to(self.root / "allowed", target_is_directory=True)
        policy = Policy.from_dict({"directories": {"allow": ["alias"], "deny": []}})
        with self.assertRaises(PolicyError):
            ToolRuntime(self.root, policy=policy)

    def test_even_internal_parent_path_is_rejected(self):
        with self.assertRaises(ToolError):
            ToolRuntime(self.root).execute("read_file", {"path": "private/../allowed/data.txt"})


@unittest.skipUnless(os.getenv("REQUIRE_SANDBOX_TESTS") == "1", "Requires native macOS sandbox")
class BashRequirementsTests(DirectoryCase):
    def test_sample_policy_has_allowed_and_denied_commands_and_directories(self):
        policy = Policy.load(Path(__file__).resolve().parents[1] / "policies/demo.json")
        (self.root / "allowed/test_example.py").write_text("import unittest\nclass Test(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n")
        runtime = ToolRuntime(self.root, policy=policy, approve=lambda request: True)
        result = runtime.execute("Bash", {"command": "python -m unittest discover -v", "cwd": "allowed"})
        self.assertTrue(result.startswith("exit_code=0\n"), result)
        for args in ({"command": "sleep 0"}, {"command": "python -m unittest discover -v", "cwd": "private"}):
            with self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
                runtime.execute("Bash", args)
        with self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
            runtime.execute("read_file", {"path": "private/data.txt"})

    def test_bash_runs_in_selected_cwd_and_separates_streams(self):
        runtime = ToolRuntime(self.root, approve=lambda request: True)
        result = runtime.execute("Bash", {"command": "cat data.txt", "cwd": "allowed"})
        self.assertTrue(result.startswith("exit_code=0\n"), result)
        self.assertIn("stdout:\npublic", result)
        self.assertIn("stderr:", result)
        (self.root / "allowed/test_output.py").write_text("import unittest\nprint('from stdout')\nclass Test(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n")
        result = runtime.execute("Bash", {"command": "python -m unittest discover -v", "cwd": "allowed"})
        stdout, stderr = result.split("stdout:\n", 1)[1].split("\nstderr:\n", 1)
        self.assertIn("from stdout", stdout)
        self.assertNotIn("Ran 1 test", stdout)
        self.assertIn("Ran 1 test", stderr)

    def test_directory_policy_also_limits_the_child_process(self):
        policy = Policy.from_dict({"directories": {"allow": ["."], "deny": ["private"]}})
        (self.root / "test_access.py").write_text("import unittest\nfrom pathlib import Path\nclass Test(unittest.TestCase):\n    def test_denied(self):\n        with self.assertRaises(PermissionError):\n            Path('private/data.txt').read_text()\n")
        result = ToolRuntime(self.root, policy=policy, approve=lambda request: True).execute("Bash", {"command": "python -m unittest discover -v"})
        self.assertTrue(result.startswith("exit_code=0\n"), result)


if __name__ == "__main__":
    unittest.main()
