from __future__ import annotations
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hw_ai_agent.tools import ToolError, ToolRuntime

class ToolRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.runtime = ToolRuntime(self.root, approve=lambda request: True)

    def tearDown(self):
        self.temporary.cleanup()

    def test_write_read_and_list(self):
        self.runtime.execute("write_file", {"path": "pkg/value.txt", "content": "hello"})
        self.assertEqual(self.runtime.execute("read_file", {"path": "pkg/value.txt"}), "hello")
        self.assertIn("pkg/value.txt", self.runtime.execute("list_files", {"recursive": True}))

    def test_parent_and_absolute_paths(self):
        for path in ("../secret", "/etc/passwd"):
            with self.subTest(path=path), self.assertRaises(ToolError):
                self.runtime.read_file(path)

    def test_external_symlink_is_blocked(self):
        (self.root / "outside").symlink_to(self.root.parent)
        with self.assertRaisesRegex(ToolError, "escapes"):
            self.runtime.write_file("outside/stolen", "x")

    def test_tests_and_aliases_are_read_only(self):
        test = self.root / "test_order.py"
        test.write_text("original")
        (self.root / "alias.py").symlink_to(test)
        for path in ("test_order.py", "alias.py", "tests/helper.py", "nested/tests/helper.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ToolError, "read-only"):
                self.runtime.write_file(path, "changed")
        self.assertEqual(test.read_text(), "original")

    def test_secrets_are_hidden(self):
        (self.root / ".env").write_text("SECRET=example")
        for tool, arguments in (("read_file", {"path": ".env"}),
                                ("write_file", {"path": ".git/config", "content": "x"})):
            with self.assertRaisesRegex(ToolError, "protected"):
                self.runtime.execute(tool, arguments)
        self.assertNotIn(".env", self.runtime.list_files(recursive=True))

    def test_invalid_arguments_and_unknown_tool(self):
        cases = [("missing", {}), ("read_file", {}), ("read_file", {"path": 3}),
                 ("list_files", {"recursive": "true"}), ("list_files", {"unexpected": 1})]
        for name, args in cases:
            with self.subTest(name=name, args=args), self.assertRaises(ToolError):
                self.runtime.execute(name, args)

    def test_command_injection_and_arbitrary_python_are_rejected(self):
        commands = ["pwd && ls", "cat $HOME/.ssh/id_rsa", "find . -delete", "python -c 'print(1)'",
                    "python -m http.server", "/tmp/python -m unittest", "pwd\nls",
                    "cat ../secret", "curl https://example.com", "ls > output"]
        for command in commands:
            with self.subTest(command=command), self.assertRaises(ToolError):
                self.runtime._argv(command)

    def test_file_limits(self):
        runtime = ToolRuntime(self.root, max_file_chars=4)
        with self.assertRaisesRegex(ToolError, "limit"):
            runtime.write_file("large", "12345")
        (self.root / "large").write_text("12345")
        with self.assertRaisesRegex(ToolError, "limit"):
            runtime.read_file("large")

    def test_bash_fails_closed_on_unsupported_system(self):
        with patch("hw_ai_agent.tools.sys.platform", "unsupported"):
            with self.assertRaisesRegex(ToolError, "no unsafe fallback"):
                self.runtime.bash("pwd")

    def test_timeout_and_output_limits_in_process_runner(self):
        # Exercise the runner with known test processes. OS protection is tested separately.
        runtime = ToolRuntime(self.root, command_timeout=0.1, max_output_chars=100)
        with patch.object(runtime, "_sandbox", return_value=[sys.executable, "-c", "import time; time.sleep(2)"]):
            with self.assertRaisesRegex(ToolError, "timed out"):
                runtime.bash("pwd")
        with patch.object(runtime, "_sandbox", return_value=[sys.executable, "-c", "print('x' * 10000)"]):
            result = runtime.bash("pwd")
            self.assertIn("output truncated", result)
            self.assertLess(len(result), 250)

@unittest.skipUnless(os.getenv("REQUIRE_SANDBOX_TESTS") == "1", "Set REQUIRE_SANDBOX_TESTS=1 for real OS sandbox tests")
class SandboxIntegrationTests(unittest.TestCase):
    setUp = ToolRuntimeTests.setUp
    tearDown = ToolRuntimeTests.tearDown

    def test_native_command_runs(self):
        self.assertTrue(self.runtime.bash("pwd").startswith("exit_code=0\n"))

    def test_test_file_cannot_be_changed_by_python(self):
        target = self.root / "test_victim.py"
        target.write_text("# unchanged\n")
        (self.root / "test_attack.py").write_text(
            "from pathlib import Path\nPath('test_victim.py').write_text('changed')\n")
        result = self.runtime.bash("python -m unittest discover -v")
        self.assertTrue(result.startswith("exit_code=1\n"), result)
        self.assertEqual(target.read_text(), "# unchanged\n")

    def test_outside_file_and_network_are_blocked(self):
        outside = self.root.parent / (self.root.name + "-secret")
        outside.write_text("private")
        (self.root / ".env").write_text("SECRET=example")
        try:
            source = f'''import os, socket, unittest
from pathlib import Path
class Isolation(unittest.TestCase):
    def test_read(self):
        with self.assertRaises(PermissionError):
            Path({str(outside)!r}).read_text()
    def test_network(self):
        with self.assertRaises(PermissionError):
            socket.create_connection(('127.0.0.1', 11434), timeout=1)
    def test_secret_file(self):
        with self.assertRaises(PermissionError):
            Path('.env').read_text()
    def test_secret_environment(self):
        self.assertNotIn('OPENROUTER_API_KEY', os.environ)
'''
            (self.root / "test_isolation.py").write_text(source)
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "example-test-secret"}):
                result = self.runtime.bash("python -m unittest discover -v")
            self.assertTrue(result.startswith("exit_code=0\n"), result)
        finally:
            outside.unlink()

if __name__ == "__main__":
    unittest.main()
