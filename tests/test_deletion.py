from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.policy import Policy
from hw_ai_agent.session import SessionStore, UncertainActionError
from hw_ai_agent.tools import ToolError, ToolRuntime
from hw_ai_agent.workflow_demo import WORKFLOW_REQUEST, WORKFLOW_SYSTEM_PROMPT, OPTIONAL_FILES, call


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)

    def complete(self, messages, tools):
        return next(self.responses)


def remove(name):
    return call(name, "delete_file", {"path": name})


class DeletionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        for name in ("a.py", "b.py", "test_keep.py", ".env"):
            (self.workspace / name).write_text("original", encoding="utf-8")
        self.store = SessionStore(self.root / ".agent/state.sqlite3")

    def tearDown(self):
        self.store.db.close()
        self.temporary.cleanup()

    def runtime(self, **kwargs):
        return ToolRuntime(self.workspace, trash_dir=self.root / ".agent/trash", **kwargs)

    def test_safe_confirms_every_deletion_even_with_allow(self):
        approve = Mock(side_effect=[True, False])
        runtime = self.runtime(approve=approve, policy=Policy.from_dict({"defaults": {"delete_file": "allow"}}))
        agent = CodingAgent(Client({"tool_calls": [remove("a.py"), remove("b.py")]}, {"content": "done"}),
                            runtime, store=self.store)
        agent.run("Optimize")
        self.assertEqual(approve.call_count, 2)
        self.assertFalse((self.workspace / "a.py").exists())
        self.assertTrue((self.workspace / "b.py").exists())
        self.assertTrue(all(c.args[0].scope == "call" for c in approve.call_args_list))

    def test_auto_confirms_every_deletion_and_preserves_recovery_copies(self):
        approve = Mock(return_value=True)
        runtime = self.runtime(mode="auto", approve=approve)
        CodingAgent(Client({"tool_calls": [remove("a.py"), remove("b.py")]}, {"content": "done"}),
                    runtime, store=self.store).run("Optimize")
        self.assertEqual(approve.call_count, 2)
        self.assertTrue(all(c.args[0].scope == "call" for c in approve.call_args_list))
        archives = list(runtime.trash_dir.iterdir())
        self.assertEqual(len(archives), 2)
        for archive in archives:
            self.assertEqual((archive / "file").read_text(), "original")
            manifest = json.loads((archive / "info.json").read_text())
            self.assertIn(manifest["path"], {"a.py", "b.py"})
            self.assertEqual(archive.stat().st_mode & 0o777, 0o700)

    def test_auto_delete_allow_still_requires_confirmation(self):
        runtime = self.runtime(mode="auto", policy=Policy.from_dict({"defaults": {"delete_file": "allow"}}))
        with self.assertRaisesRegex(ToolError, "APPROVAL_REQUIRED"):
            runtime.execute("delete_file", {"path": "a.py"})
        self.assertTrue((self.workspace / "a.py").exists())

    def test_safe_allow_does_not_skip_write_or_bash_confirmation(self):
        approve = Mock(return_value=True)
        runtime = self.runtime(approve=approve, policy=Policy.from_dict({"defaults": {"write_file": "allow", "Bash": "allow"}}))
        runtime.bash = Mock(return_value="exit_code=0")
        runtime.execute("write_file", {"path": "a.py", "content": "new"})
        runtime.execute("Bash", {"command": "pwd"})
        self.assertEqual(approve.call_count, 2)

    def test_auto_refused_deletion_is_not_cached(self):
        approve = Mock(return_value=False)
        runtime = self.runtime(mode="auto", approve=approve)
        CodingAgent(Client({"tool_calls": [remove("a.py"), remove("b.py")]}, {"content": "denied"}),
                    runtime, store=self.store).run("Optimize")
        self.assertEqual(approve.call_count, 2)
        self.assertEqual(runtime.session_permissions, set())

    def test_auto_resume_requires_fresh_deletion_confirmation(self):
        def stop(event):
            if event["event"] == "tool_result":
                raise KeyboardInterrupt
        first = CodingAgent(Client({"tool_calls": [remove("a.py"), remove("b.py")]}),
                            self.runtime(mode="auto", approve=lambda request: True), store=self.store, on_event=stop)
        with self.assertRaises(KeyboardInterrupt):
            first.run("Optimize")
        approve = Mock(return_value=False)
        CodingAgent(Client({"content": "done"}), self.runtime(mode="auto", approve=approve), store=self.store).resume(first.session_id)
        approve.assert_called_once()
        self.assertTrue((self.workspace / "b.py").exists())

    def test_new_auto_session_requires_new_deletion_permission(self):
        approve = Mock(return_value=True)
        runtime = self.runtime(mode="auto", approve=approve)
        for name in ("a.py", "b.py"):
            CodingAgent(Client({"tool_calls": [remove(name)]}, {"content": "done"}), runtime, store=self.store).run("Optimize")
        self.assertEqual(approve.call_count, 2)

    def test_policy_deny_overrides_cached_deletion_permission(self):
        runtime = self.runtime(mode="auto", policy=Policy.from_dict({"defaults": {"delete_file": "deny"}}))
        runtime.session_permissions = {"delete_file"}
        with self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
            runtime.execute("delete_file", {"path": "a.py"})
        self.assertTrue((self.workspace / "a.py").exists())

    def test_tests_secrets_directories_and_external_paths_cannot_be_deleted(self):
        runtime = self.runtime(mode="auto", approve=lambda request: True)
        for path in ("test_keep.py", ".env", ".", "../outside.py"):
            with self.subTest(path=path), self.assertRaises(ToolError):
                runtime.execute("delete_file", {"path": path})
        self.assertTrue((self.workspace / "test_keep.py").exists())
        self.assertTrue((self.workspace / ".env").exists())

    def test_symlinks_and_hardlinks_cannot_be_deleted(self):
        alias = self.workspace / "alias.py"
        alias.symlink_to(self.workspace / "a.py")
        os.link(self.workspace / "b.py", self.workspace / "hard.py")
        runtime = self.runtime(mode="auto", approve=lambda request: True)
        for path in ("alias.py", "hard.py"):
            with self.subTest(path=path), self.assertRaises(ToolError):
                runtime.execute("delete_file", {"path": path})
        self.assertTrue(alias.is_symlink())
        self.assertTrue((self.workspace / "a.py").exists())

    def test_file_change_during_confirmation_prevents_deletion(self):
        def approve(request):
            (self.workspace / "a.py").write_text("operator edit")
            return True
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            self.runtime(approve=approve).execute("delete_file", {"path": "a.py"})
        self.assertEqual((self.workspace / "a.py").read_text(), "operator edit")

    def test_replacement_at_rename_boundary_is_preserved(self):
        original_rename = os.rename
        def race(source, destination, **kwargs):
            original_rename(self.workspace / "a.py", self.workspace / "original-preserved.py")
            (self.workspace / "a.py").write_text("replacement")
            original_rename(source, destination, **kwargs)
        runtime = self.runtime(approve=lambda request: True)
        with patch("hw_ai_agent.filesystem.os.rename", side_effect=race), self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            runtime.execute("delete_file", {"path": "a.py"})
        self.assertEqual((self.workspace / "original-preserved.py").read_text(), "original")
        self.assertEqual((self.workspace / "a.py").read_text(), "replacement")
        self.assertEqual(next(runtime.trash_dir.iterdir()).joinpath("file").read_text(), "replacement")

    def test_missing_file_retry_is_idempotent(self):
        runtime = self.runtime(mode="auto", approve=lambda request: True)
        first = runtime.execute("delete_file", {"path": "a.py"})
        second = runtime.execute("delete_file", {"path": "a.py"})
        self.assertIn("recovery_id=", first)
        self.assertEqual(second, "already absent: a.py")
        self.assertEqual(len(list(runtime.trash_dir.iterdir())), 1)

    def test_uncertain_deletion_blocks_until_explicit_resolution(self):
        runtime = self.runtime(mode="auto", approve=lambda request: True)
        original = runtime.execute
        def execute_then_stop(name, arguments):
            original(name, arguments)
            raise KeyboardInterrupt
        runtime.execute = execute_then_stop
        agent = CodingAgent(Client({"tool_calls": [remove("a.py")]}), runtime, store=self.store)
        with self.assertRaises(KeyboardInterrupt):
            agent.run("Optimize")
        with self.assertRaises(UncertainActionError):
            CodingAgent(Client(), self.runtime(mode="auto"), store=self.store).resume(agent.session_id)
        self.store.resolve(agent.session_id, "1:0", "retry")
        approve = Mock(return_value=True)
        resumed = CodingAgent(Client({"content": "done"}), self.runtime(mode="auto", approve=approve), store=self.store)
        resumed.resume(agent.session_id)
        approve.assert_called_once()
        self.assertEqual(approve.call_args.args[0].scope, "call")
        self.assertIn("already absent", resumed.messages[-2]["content"])

    def test_recovery_storage_cannot_be_inside_workspace(self):
        with self.assertRaises(ValueError):
            ToolRuntime(self.workspace, trash_dir=self.workspace / "trash")

    def test_legacy_policy_cannot_gain_deletion_permission_on_restore(self):
        policy = Policy.restore({"defaults": {"write_file": "allow", "Bash": "allow"}, "rules": []})
        self.assertEqual(policy.decide("delete_file").action.value, "deny")

    def test_prompt_does_not_name_cleanup_candidates_or_request_deletion(self):
        for filename in OPTIONAL_FILES:
            self.assertNotIn(filename, WORKFLOW_REQUEST)
            self.assertNotIn(filename, WORKFLOW_SYSTEM_PROMPT)
        self.assertIn("optimize", WORKFLOW_REQUEST)
        self.assertIn("remove files", WORKFLOW_REQUEST)
        self.assertNotIn("unused", WORKFLOW_REQUEST)


@unittest.skipUnless(os.getenv("REQUIRE_SANDBOX_TESTS") == "1", "Requires native macOS sandbox")
class ExpandedWorkflowTests(unittest.TestCase):
    def run_expanded(self, mode, remove_files=True):
        from types import SimpleNamespace
        from hw_ai_agent.workflow_demo import run_workflow_demo, FIXED_TAX, FIXED_RECEIPT
        class ModelClient(Client):
            settings = SimpleNamespace(provider="ollama", model="test-double", base_url="http://localhost:11434/v1")
            usage = []
            max_tokens = 2048
        actions = [call("tax", "write_file", {"path": "tax.py", "content": FIXED_TAX}),
                   call("receipt", "write_file", {"path": "receipt.py", "content": FIXED_RECEIPT})]
        if remove_files:
            actions.extend(remove(name) for name in OPTIONAL_FILES[:2])
        actions.extend([call("ls", "Bash", {"command": "ls"}),
                        call("tests", "Bash", {"command": "python -m unittest discover -v"})])
        client = ModelClient({"tool_calls": actions}, {"content": "done"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approve = Mock(return_value=True)
            report = run_workflow_demo(root / "workspace", root / ".agent/state.sqlite3",
                                       mode=mode, approve=approve, client=client)
            self.assertIn("Ran 4 tests", report["verification"])
            self.assertTrue(report["tests_unchanged"])
            self.assertTrue((root / "workspace/formatting.py").exists())
            return report, approve.call_count

    def test_expanded_workflow_in_both_modes(self):
        for mode in ("safe", "auto"):
            with self.subTest(mode=mode):
                report, prompts = self.run_expanded(mode)
                self.assertTrue(report["success"])
                self.assertEqual(len(report["removed_optional_files"]), 2)
                self.assertEqual(prompts, 6 if mode == "safe" else 2)
                self.assertEqual(report["permission_requests"],
                                 {"write_file": 2, "delete_file": 2, "Bash": 2} if mode == "safe" else {"delete_file": 2})

    def test_functional_pass_without_cleanup_is_not_full_demo_pass(self):
        report, _ = self.run_expanded("auto", remove_files=False)
        self.assertTrue(report["functional_success"])
        self.assertFalse(report["cleanup_complete"])
        self.assertFalse(report["success"])


if __name__ == "__main__":
    unittest.main()
