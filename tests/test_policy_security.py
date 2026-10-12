from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hw_ai_agent.policy import Policy
from hw_ai_agent.tools import ToolError, ToolRuntime


class ApprovalTargetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.safe = self.root / "order.py"
        self.private = self.root / "private.py"
        self.safe.write_text("safe original")
        self.private.write_text("private original")
        self.policy = Policy.from_dict({"rules": [
            {"tool": "write_file", "path": "private.py", "action": "deny"}]})

    def tearDown(self):
        self.temporary.cleanup()

    def test_symlink_retarget_during_approval_is_blocked(self):
        alias = self.root / "alias.py"
        alias.symlink_to(self.safe)
        def approve(request):
            alias.unlink()
            alias.symlink_to(self.private)
            return True
        runtime = ToolRuntime(self.root, policy=self.policy, approve=approve)
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            runtime.execute("write_file", {"path": "alias.py", "content": "approved"})
        self.assertEqual(self.safe.read_text(), "safe original")
        self.assertEqual(self.private.read_text(), "private original")
        self.assertEqual(runtime.policy_events[-1]["outcome"], "blocked")

    def test_file_replaced_with_symlink_during_approval_is_blocked(self):
        def approve(request):
            self.safe.unlink()
            self.safe.symlink_to(self.private)
            return True
        runtime = ToolRuntime(self.root, policy=self.policy, approve=approve)
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            runtime.execute("write_file", {"path": "order.py", "content": "approved"})
        self.assertEqual(self.private.read_text(), "private original")

    def test_parent_retarget_during_approval_is_blocked(self):
        folder = self.root / "pkg"
        folder.mkdir()
        (folder / "order.py").write_text("original")
        private_folder = self.root / "private"
        private_folder.mkdir()
        (private_folder / "order.py").write_text("private original")
        def approve(request):
            folder.rename(self.root / "old-pkg")
            folder.symlink_to(private_folder, target_is_directory=True)
            return True
        runtime = ToolRuntime(self.root, approve=approve)
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            runtime.execute("write_file", {"path": "pkg/order.py", "content": "approved"})
        self.assertEqual((private_folder / "order.py").read_text(), "private original")
        self.assertEqual((self.root / "old-pkg/order.py").read_text(), "original")

    def test_replaced_parent_at_same_path_is_blocked(self):
        folder = self.root / "pkg"
        folder.mkdir()
        (folder / "order.py").write_text("original")
        def approve(request):
            folder.rename(self.root / "old-pkg")
            folder.mkdir()
            (folder / "order.py").write_text("replacement")
            return True
        runtime = ToolRuntime(self.root, approve=approve)
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            runtime.execute("write_file", {"path": "pkg/order.py", "content": "approved"})
        self.assertEqual((folder / "order.py").read_text(), "replacement")

    def test_edit_while_approval_pending_is_preserved(self):
        def approve(request):
            self.safe.write_text("operator edit")
            return True
        runtime = ToolRuntime(self.root, approve=approve)
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            runtime.execute("write_file", {"path": "order.py", "content": "approved"})
        self.assertEqual(self.safe.read_text(), "operator edit")

    def test_new_file_appearing_while_approval_pending_is_preserved(self):
        new_file = self.root / "new.py"
        def approve(request):
            new_file.write_text("operator file")
            return True
        runtime = ToolRuntime(self.root, approve=approve)
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            runtime.execute("write_file", {"path": "new.py", "content": "approved"})
        self.assertEqual(new_file.read_text(), "operator file")

    def test_replacement_at_same_path_before_approval_finishes_is_preserved(self):
        def approve(request):
            self.safe.rename(self.root / "old-order.py")
            self.safe.write_text("replacement")
            return True
        runtime = ToolRuntime(self.root, approve=approve)
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            runtime.execute("write_file", {"path": "order.py", "content": "approved"})
        self.assertEqual(self.safe.read_text(), "replacement")
        self.assertEqual((self.root / "old-order.py").read_text(), "safe original")

    def test_symlink_swap_at_final_open_does_not_touch_denied_file(self):
        original_open = os.open
        def swap_then_open(path, flags, *args, **kwargs):
            if path == "order.py" and flags & (os.O_WRONLY | os.O_RDWR):
                self.safe.unlink()
                self.safe.symlink_to(self.private)
            return original_open(path, flags, *args, **kwargs)
        runtime = ToolRuntime(self.root, approve=lambda request: True)
        with patch("hw_ai_agent.filesystem.os.open", side_effect=swap_then_open):
            with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
                runtime.execute("write_file", {"path": "order.py", "content": "approved"})
        self.assertEqual(self.private.read_text(), "private original")

    def test_hardlink_added_during_approval_is_blocked(self):
        def approve(request):
            os.link(self.safe, self.root / "test_alias.py")
            return True
        runtime = ToolRuntime(self.root, approve=approve)
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            runtime.execute("write_file", {"path": "order.py", "content": "approved"})
        self.assertEqual(self.safe.read_text(), "safe original")
        self.assertEqual((self.root / "test_alias.py").read_text(), "safe original")

    def test_unchanged_approved_alias_writes_checked_target(self):
        alias = self.root / "alias.py"
        alias.symlink_to(self.safe)
        runtime = ToolRuntime(self.root, approve=lambda request: True)
        runtime.execute("write_file", {"path": "alias.py", "content": "approved"})
        self.assertEqual(self.safe.read_text(), "approved")
        self.assertTrue(alias.is_symlink())

    def test_unsupported_bash_read_pattern_fails_before_process_start(self):
        policy = Policy.from_dict({"rules": [
            {"tool": "read_file", "path": "private/[ab].txt", "action": "deny"}]})
        runtime = ToolRuntime(self.root, policy=policy, mode="auto", approve=lambda request: True)
        with patch("hw_ai_agent.tools.sys.platform", "darwin"), \
             patch("hw_ai_agent.tools.Path.exists", return_value=True), \
             patch("hw_ai_agent.tools.subprocess.Popen") as process:
            with self.assertRaisesRegex(ToolError, "POLICY_DENIED"):
                runtime.execute("Bash", {"command": "pwd"})
        process.assert_not_called()


class ReadTargetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "public.txt").write_text("public")
        self.private = self.root / "private.txt"
        self.private.write_text("outside workspace")
        self.runtime = ToolRuntime(self.workspace)

    def tearDown(self):
        self.temporary.cleanup()

    def test_leaf_swapped_to_external_symlink_at_open_is_blocked(self):
        original_open = os.open
        def swap(path, flags, *args, **kwargs):
            if path == "public.txt":
                file = self.workspace / "public.txt"
                file.unlink()
                file.symlink_to(self.private)
            return original_open(path, flags, *args, **kwargs)
        with patch("hw_ai_agent.filesystem.os.open", side_effect=swap):
            with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
                self.runtime.execute("read_file", {"path": "public.txt"})

    def test_parent_swapped_to_external_symlink_at_open_is_blocked(self):
        nested = self.workspace / "nested"
        nested.mkdir()
        (nested / "public.txt").write_text("public")
        original_open = os.open
        def swap(path, flags, *args, **kwargs):
            if path == "nested":
                nested.rename(self.workspace / "old-nested")
                nested.symlink_to(self.root, target_is_directory=True)
            return original_open(path, flags, *args, **kwargs)
        with patch("hw_ai_agent.filesystem.os.open", side_effect=swap):
            with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
                self.runtime.execute("read_file", {"path": "nested/public.txt"})

    def test_parent_replaced_after_open_cannot_redirect_the_read(self):
        nested = self.workspace / "nested"
        nested.mkdir()
        (nested / "private.txt").write_text("public")
        original_open = os.open
        def swap(path, flags, *args, **kwargs):
            if path == "private.txt":
                nested.rename(self.workspace / "old-nested")
                nested.symlink_to(self.root, target_is_directory=True)
            return original_open(path, flags, *args, **kwargs)
        with patch("hw_ai_agent.filesystem.os.open", side_effect=swap):
            self.assertEqual(self.runtime.execute("read_file", {"path": "nested/private.txt"}), "public")

    def test_external_hardlink_is_not_readable(self):
        os.link(self.private, self.workspace / "alias.txt")
        with self.assertRaisesRegex(ToolError, "TARGET_CHANGED"):
            self.runtime.execute("read_file", {"path": "alias.txt"})

    def test_recursive_listing_cannot_follow_a_swapped_directory(self):
        nested = self.workspace / "nested"
        nested.mkdir()
        private_dir = self.root / "private"
        private_dir.mkdir()
        (private_dir / "outside-canary.txt").write_text("private")
        original_open = os.open
        def swap(path, flags, *args, **kwargs):
            if path == "nested" and not nested.is_symlink():
                nested.rename(self.workspace / "old-nested")
                nested.symlink_to(private_dir, target_is_directory=True)
            return original_open(path, flags, *args, **kwargs)
        with patch("hw_ai_agent.filesystem.os.open", side_effect=swap):
            result = self.runtime.execute("list_files", {"recursive": True})
        self.assertNotIn("outside-canary", result)

    def test_applied_verification_cannot_follow_a_swapped_symlink(self):
        from hw_ai_agent.cli import verify_applied
        (self.workspace / "public.txt").write_text("unapplied")
        self.private.write_text("expected")
        item = {"call": {"function": {"arguments": json.dumps({"path": "public.txt", "content": "expected"})}}}
        original_open = os.open
        def swap(path, flags, *args, **kwargs):
            if path == "public.txt":
                file = self.workspace / "public.txt"
                file.unlink()
                file.symlink_to(self.private)
            return original_open(path, flags, *args, **kwargs)
        with patch("hw_ai_agent.filesystem.os.open", side_effect=swap):
            self.assertFalse(verify_applied(self.runtime, item))


@unittest.skipUnless(os.getenv("REQUIRE_SANDBOX_TESTS") == "1", "Requires native macOS sandbox")
class BashFilesystemPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def run_checks(self, runtime, source):
        (self.root / "test_checks.py").write_text(source, encoding="utf-8")
        result = runtime.execute("Bash", {"command": "python -m unittest discover -v"})
        self.assertTrue(result.startswith("exit_code=0\n"), result)

    def test_preapproved_order_import_cannot_create_another_file(self):
        policy = Policy.from_dict({"defaults": {"write_file": "deny"}, "rules": [
            {"tool": "write_file", "path": "order.py", "action": "allow"}]})
        runtime = ToolRuntime(self.root, policy=policy, mode="auto", approve=lambda request: True)
        runtime.execute("write_file", {"path": "order.py", "content":
            "from pathlib import Path\nPath('other.py').write_text('bypassed')\n"})
        self.run_checks(runtime, """import unittest
class Check(unittest.TestCase):
    def test_import_cannot_write(self):
        with self.assertRaises(PermissionError):
            __import__('order')
""")
        self.assertFalse((self.root / "other.py").exists())

    def test_bash_cannot_overwrite_even_an_allowed_file(self):
        file = self.root / "order.py"
        file.write_text("original")
        policy = Policy.from_dict({"rules": [{"tool": "write_file", "action": "allow"}]})
        runtime = ToolRuntime(self.root, policy=policy, mode="auto", approve=lambda request: True)
        self.run_checks(runtime, """import unittest
from pathlib import Path
class Check(unittest.TestCase):
    def test_cannot_write(self):
        with self.assertRaises(PermissionError):
            Path('order.py').write_text('bypassed')
""")
        self.assertEqual(file.read_text(), "original")

    def test_bash_cannot_read_private_files_denied_by_policy(self):
        private = self.root / "private"
        private.mkdir()
        (private / "data.txt").write_text("private data")
        policy = Policy.from_dict({"rules": [
            {"tool": "read_file", "path": "private/**", "action": "deny"}]})
        runtime = ToolRuntime(self.root, policy=policy, mode="auto", approve=lambda request: True)
        self.run_checks(runtime, """import unittest
from pathlib import Path
class Check(unittest.TestCase):
    def test_cannot_read(self):
        with self.assertRaises(PermissionError):
            Path('private/data.txt').read_text()
""")

    def test_ask_read_rule_cannot_be_implicitly_approved_by_bash(self):
        (self.root / "sensitive.txt").write_text("private")
        policy = Policy.from_dict({"rules": [
            {"tool": "read_file", "path": "sensitive.txt", "action": "ask"}]})
        runtime = ToolRuntime(self.root, policy=policy, approve=lambda request: True)
        self.run_checks(runtime, """import unittest
from pathlib import Path
class Check(unittest.TestCase):
    def test_cannot_read(self):
        with self.assertRaises(PermissionError):
            Path('sensitive.txt').read_text()
""")

    def test_denied_read_cannot_bypass_policy_with_alias_or_newline_name(self):
        private = self.root / "private"
        private.mkdir()
        (private / "data.txt").write_text("private")
        (private / "data\n.txt").write_text("private")
        (self.root / "alias.txt").symlink_to(private / "data.txt")
        policy = Policy.from_dict({"rules": [
            {"tool": "read_file", "path": "private/**", "action": "deny"}]})
        runtime = ToolRuntime(self.root, policy=policy, mode="auto", approve=lambda request: True)
        self.run_checks(runtime, """import unittest
from pathlib import Path
class Check(unittest.TestCase):
    def test_denied_paths(self):
        for name in ['alias.txt', 'private/data\\n.txt']:
            with self.subTest(name=name), self.assertRaises(PermissionError):
                Path(name).read_text()
""")

    def test_exact_read_allow_does_not_allow_other_files(self):
        (self.root / "allowed.txt").write_text("allowed")
        (self.root / "denied.txt").write_text("denied")
        policy = Policy.from_dict({"defaults": {"read_file": "deny"}, "rules": [
            {"tool": "read_file", "path": ".", "action": "allow"},
            {"tool": "read_file", "path": "test_checks.py", "action": "allow"},
            {"tool": "read_file", "path": "allowed.txt", "action": "allow"}]})
        runtime = ToolRuntime(self.root, policy=policy, mode="auto", approve=lambda request: True)
        self.run_checks(runtime, """import unittest
from pathlib import Path
class Check(unittest.TestCase):
    def test_read_permissions(self):
        self.assertEqual(Path('allowed.txt').read_text(), 'allowed')
        with self.assertRaises(PermissionError):
            Path('denied.txt').read_text()
""")

    def test_original_demo_still_passes_with_scoped_policy(self):
        from hw_ai_agent.config import Settings
        from hw_ai_agent.demo import run_demo
        class ScriptedClient:
            settings = Settings("openrouter", "test-model", "https://example.invalid/v1", "test")
            usage = []
            def __init__(self):
                code = ("def calculate_total(prices, discount_percent):\n"
                        "    if not 0 <= discount_percent <= 100:\n"
                        "        raise ValueError('invalid discount')\n"
                        "    return sum(prices) * (1 - discount_percent / 100)\n")
                self.responses = iter([
                    {"tool_calls": [
                        {"id": "write", "function": {"name": "write_file", "arguments":
                            json.dumps({"path": "order.py", "content": code})}},
                        {"id": "test", "function": {"name": "Bash", "arguments":
                            json.dumps({"command": "python -m unittest discover -v"})}}]},
                    {"content": "Tests pass."},
                ])
            def complete(self, messages, tools):
                return next(self.responses)
        policy = Policy.load(Path(__file__).resolve().parents[1] / "policies/demo.json")
        report = run_demo(ScriptedClient(), self.root / "demo", policy=policy, mode="auto", approve=lambda request: True)
        self.assertTrue(report["success"])
        self.assertTrue(report["tests_unchanged"])
        self.assertTrue(report["model_ran_passing_tests"])
        self.assertIn("Ran 6 tests", report["verification"])


if __name__ == "__main__":
    unittest.main()
