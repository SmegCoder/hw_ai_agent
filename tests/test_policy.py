from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.policy import Action, Policy, PolicyError
from hw_ai_agent.policy_demo import run_policy_demo
from hw_ai_agent.tools import ToolError, ToolRuntime


class PolicyTests(unittest.TestCase):
    def test_safe_defaults(self):
        policy = Policy()
        self.assertEqual(policy.decide("read_file").action, Action.ALLOW)
        self.assertEqual(policy.decide("write_file").action, Action.ASK)
        self.assertEqual(policy.decide("unknown").action, Action.DENY)

    def test_deny_wins_in_every_rule_order(self):
        rules = [{"tool": "write_file", "action": action} for action in ("allow", "ask", "deny")]
        for ordering in itertools.permutations(rules):
            with self.subTest(ordering=ordering):
                policy = Policy.from_dict({"rules": list(ordering)})
                self.assertEqual(policy.decide("write_file", path="order.py").action, Action.DENY)

    def test_ask_wins_over_allow(self):
        policy = Policy.from_dict({"rules": [{"tool": "*", "action": "allow"},
                                            {"tool": "write_file", "action": "ask"}]})
        self.assertEqual(policy.decide("write_file").action, Action.ASK)

    def test_specific_allow_can_override_default_deny(self):
        policy = Policy.from_dict({"defaults": {"write_file": "deny"}, "rules": [
            {"tool": "write_file", "path": "order.py", "action": "allow"}]})
        self.assertEqual(policy.decide("write_file", path="order.py").action, Action.ALLOW)
        self.assertEqual(policy.decide("write_file", path="other.py").action, Action.DENY)

    def test_command_patterns_match_the_entire_command(self):
        policy = Policy.from_dict({"defaults": {"Bash": "deny"}, "rules": [
            {"tool": "Bash", "command": "python -m unittest discover -v", "action": "allow"}]})
        self.assertEqual(policy.decide("Bash", command="python -m unittest discover -v").action, Action.ALLOW)
        self.assertEqual(policy.decide("Bash", command="python -m unittest discover -v extra").action, Action.DENY)

    def test_invalid_configurations_fail_closed(self):
        cases = [[], {"unknown": 1}, {"defaults": {"missing": "allow"}},
                 {"defaults": {"write_file": "approve"}}, {"rules": {}},
                 {"rules": [{"tool": "write_file"}]},
                 {"rules": [{"tool": "write_file", "action": "allow", "path": "../outside"}]},
                 {"rules": [{"tool": "Bash", "action": "allow", "path": "order.py"}]},
                 {"rules": [{"tool": "write_file", "action": "allow", "command": "pwd"}]},
                 {"rules": [{"tool": "*", "action": "allow", "unexpected": 1}]}]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(PolicyError):
                Policy.from_dict(case)

    def test_policy_file_validation_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "policy.json"
            for text in ("not json", "x" * 64_001, '{"defaults":{"write_file":"deny","write_file":"allow"}}'):
                file.write_text(text)
                with self.subTest(text=text[:50]), self.assertRaises(PolicyError):
                    Policy.load(file)
            file.write_text(json.dumps({"defaults": {"write_file": "deny"}}))
            self.assertEqual(Policy.load(file).decide("write_file").action, Action.DENY)

    def test_offline_policy_demo_passes(self):
        report = run_policy_demo()
        self.assertTrue(report["success"])
        self.assertTrue(all(report["checks"].values()))


class RuntimePolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "order.py").write_text("original")

    def tearDown(self):
        self.directory.cleanup()

    def test_default_write_requires_approval(self):
        runtime = ToolRuntime(self.root)
        with self.assertRaisesRegex(ToolError, "APPROVAL_REQUIRED"):
            runtime.execute("write_file", {"path": "order.py", "content": "changed"})
        self.assertEqual((self.root / "order.py").read_text(), "original")
        self.assertEqual(runtime.policy_events[-1]["outcome"], "refused")

    def test_approval_shows_diff_and_allows_exact_request(self):
        approve = Mock(return_value=True)
        runtime = ToolRuntime(self.root, approve=approve)
        runtime.execute("write_file", {"path": "order.py", "content": "changed"})
        request = approve.call_args.args[0]
        self.assertIn("original", request.preview)
        self.assertIn("changed", request.preview)
        self.assertEqual(request.target, "order.py")
        self.assertEqual(len(request.fingerprint), 64)
        self.assertEqual((self.root / "order.py").read_text(), "changed")

    def test_callback_must_explicitly_return_true(self):
        for result in (False, None, 1, "yes"):
            with self.subTest(result=result):
                runtime = ToolRuntime(self.root, approve=lambda request: result)
                with self.assertRaisesRegex(ToolError, "APPROVAL_REQUIRED"):
                    runtime.execute("write_file", {"path": "order.py", "content": "changed"})
        self.assertEqual((self.root / "order.py").read_text(), "original")

    def test_broken_callback_does_not_authorize_write(self):
        runtime = ToolRuntime(self.root, approve=Mock(side_effect=RuntimeError("broken")))
        with self.assertRaisesRegex(ToolError, "APPROVAL_REQUIRED"):
            runtime.execute("write_file", {"path": "order.py", "content": "changed"})

    def test_standalone_runtime_approval_is_per_call(self):
        approve = Mock(side_effect=[True, False])
        runtime = ToolRuntime(self.root, approve=approve)
        runtime.execute("write_file", {"path": "order.py", "content": "first"})
        with self.assertRaisesRegex(ToolError, "APPROVAL_REQUIRED"):
            runtime.execute("write_file", {"path": "order.py", "content": "second"})
        self.assertEqual(approve.call_count, 2)
        self.assertNotEqual(approve.call_args_list[0].args[0].fingerprint,
                            approve.call_args_list[1].args[0].fingerprint)
        self.assertEqual((self.root / "order.py").read_text(), "first")

    def test_callback_cannot_change_original_arguments(self):
        arguments = {"path": "order.py", "content": "approved"}
        def approve(request):
            arguments["content"] = "not-approved"
            return True
        ToolRuntime(self.root, approve=approve).execute("write_file", arguments)
        self.assertEqual((self.root / "order.py").read_text(), "approved")

    def test_deny_does_not_prompt_or_run_handler(self):
        approve = Mock(return_value=True)
        runtime = ToolRuntime(self.root, approve=approve,
                              policy=Policy.from_dict({"defaults": {"write_file": "deny", "Bash": "deny"}}))
        with patch.object(runtime, "bash") as bash:
            for tool, arguments in (("write_file", {"path": "order.py", "content": "changed"}),
                                    ("bash", {"command": "pwd"})):
                with self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
                    runtime.execute(tool, arguments)
            bash.assert_not_called()
        approve.assert_not_called()
        self.assertEqual((self.root / "order.py").read_text(), "original")

    def test_allow_rules_cannot_override_runtime_boundaries(self):
        approve = Mock(return_value=True)
        runtime = ToolRuntime(self.root, approve=approve,
                              policy=Policy.from_dict({"rules": [{"tool": "*", "action": "allow"}]}))
        cases = [("write_file", {"path": "test_order.py", "content": "changed"}),
                 ("read_file", {"path": ".env"}), ("read_file", {"path": "../secret"}),
                 ("Bash", {"command": "pwd && ls"})]
        for tool, arguments in cases:
            with self.subTest(tool=tool, arguments=arguments), self.assertRaises(ToolError):
                runtime.execute(tool, arguments)
        approve.assert_not_called()
        self.assertTrue(all(e["source"] == "runtime_boundary" for e in runtime.policy_events))

    def test_rules_use_canonical_paths_and_not_symlink_aliases(self):
        (self.root / "alias.py").symlink_to(self.root / "order.py")
        runtime = ToolRuntime(self.root, policy=Policy.from_dict({"rules": [
            {"tool": "read_file", "path": "order.py", "action": "deny"}]}))
        for path in ("alias.py", "./order.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
                runtime.execute("read_file", {"path": path})

    def test_readonly_policy_prevents_changes(self):
        runtime = ToolRuntime(self.root, policy=Policy.from_dict({"defaults": {"write_file": "deny", "Bash": "deny"}}))
        self.assertEqual(runtime.execute("read_file", {"path": "order.py"}), "original")
        with self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
            runtime.execute("write_file", {"path": "new.py", "content": "x"})
        self.assertFalse((self.root / "new.py").exists())

    def test_agent_receives_refusal_and_can_finish(self):
        class Client:
            def __init__(self):
                self.messages = None
            def complete(self, messages, tools):
                self.messages = list(messages)
                if messages[-1]["role"] == "user":
                    return {"tool_calls": [{"id": "write", "function": {"name": "write_file", "arguments":
                        json.dumps({"path": "order.py", "content": "changed"})}}]}
                return {"content": "Operator refused; file preserved."}
        client = Client()
        result = CodingAgent(client, ToolRuntime(self.root, approve=lambda request: False)).run("Change file")
        self.assertIn("refused", result)
        self.assertIn("APPROVAL_REQUIRED", client.messages[-1]["content"])
        self.assertEqual((self.root / "order.py").read_text(), "original")


class TerminalApprovalTests(unittest.TestCase):
    def test_noninteractive_input_never_prompts(self):
        from hw_ai_agent.cli import confirm_action
        from hw_ai_agent.policy import ApprovalDeferred, ApprovalRequest
        request = ApprovalRequest("write_file", "order.py", "ask", "diff", "fingerprint")
        with patch("hw_ai_agent.cli.sys.stdin.isatty", return_value=False), patch("builtins.input") as prompt:
            with self.assertRaises(ApprovalDeferred):
                confirm_action(request)
        prompt.assert_not_called()

    def test_policy_inside_workspace_is_rejected(self):
        from hw_ai_agent.cli import load_policy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(PolicyError, "outside"):
                load_policy(root / "policy.json", root)


if __name__ == "__main__":
    unittest.main()
