from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hw_ai_agent.demo import BROKEN_ORDER, ORDER_TESTS, verify_demo
from hw_ai_agent.reporting import public_report


class PublicReportTests(unittest.TestCase):
    def test_paths_are_cleaned_in_nested_fields_and_keys(self):
        workspace = Path("/Users/ExampleUser/project/workspace/demo-123")
        report = {"workspace": str(workspace),
                  "baseline": f'File "{workspace}/test_order.py", line 6',
                  "events": [{"content": f"Read {workspace}/order.py",
                              "arguments": json.dumps({"path": f"{workspace}/order.py"})}],
                  "error": f"Cannot open {workspace}/missing.py",
                  "extra": {str(workspace / "order.py"): None}}
        cleaned = public_report(report, workspace)
        self.assertNotIn("ExampleUser", json.dumps(cleaned))
        self.assertEqual(cleaned["workspace"], "workspace/demo")
        self.assertIn('File "workspace/demo/test_order.py"', cleaned["baseline"])
        self.assertEqual(json.loads(cleaned["events"][0]["arguments"])["path"],
                         "workspace/demo/order.py")
        self.assertIn("workspace/demo/order.py", cleaned["extra"])

    def test_home_paths_outside_workspace_are_anonymized(self):
        report = {"error": '/Users/OtherUser/project/file.py /home/LinuxUser/private.txt',
                  "windows": r"C:\Users\WindowsUser\project\file.py",
                  "arguments": json.dumps({"path": r"C:\Users\WindowsUser\project\file.py"}),
                  "current_home": str(Path.home() / "private.txt")}
        cleaned = public_report(report, Path("/example/demo"))
        self.assertEqual(cleaned["error"], "<home>/project/file.py <home>/private.txt")
        self.assertEqual(cleaned["windows"], r"<home>\project\file.py")
        self.assertNotIn("WindowsUser", cleaned["arguments"])
        self.assertEqual(cleaned["current_home"], "<home>/private.txt")

    def test_personal_macos_temporary_paths_are_anonymized(self):
        report = {"error": "/private/var/folders/ab/user-token/T/run/file.py",
                  "event": "/var/folders/ab/user-token/C/cache.py"}
        cleaned = public_report(report, Path("/example/demo"))
        self.assertEqual(cleaned["error"], "<temp>/run/file.py")
        self.assertEqual(cleaned["event"], "<temp>/cache.py")

    def test_results_urls_and_relative_paths_are_preserved(self):
        report = {"success": True, "error": None, "usage": [{"cost": 0.012}],
                  "provider": "ollama", "model": "qwen3:4b-instruct",
                  "url": "https://example.org/docs/report.json",
                  "diff": "--- before/order.py\n+++ after/order.py\n",
                  "verification": "exit_code=0\nRan 6 tests\nOK"}
        cleaned = public_report(report, Path("/example/demo"))
        for key in report:
            self.assertEqual(cleaned[key], report[key])

    def test_private_report_is_not_mutated(self):
        workspace = Path("/Users/ExampleUser/demo")
        report = {"workspace": str(workspace), "events": [{"content": str(workspace)}]}
        cleaned = public_report(report, workspace)
        self.assertEqual(report["workspace"], str(workspace))
        self.assertEqual(report["events"][0]["content"], str(workspace))
        cleaned["events"][0]["content"] = "changed"
        self.assertEqual(report["events"][0]["content"], str(workspace))

    def test_demo_saves_clean_report_but_retains_operational_path(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            (workspace / "order.py").write_text(BROKEN_ORDER, encoding="utf-8")
            (workspace / "test_order.py").write_text(ORDER_TESTS, encoding="utf-8")
            test_hash = hashlib.sha256((workspace / "test_order.py").read_bytes()).hexdigest()
            output = f'exit_code=0\nFile "{workspace}/test_order.py"\nRan 6 tests\nOK'
            client = SimpleNamespace(settings=SimpleNamespace(provider="ollama", model="example"), usage=[])
            runtime = SimpleNamespace(bash=lambda command: output, policy_events=[])
            events = [{"event": "tool_result", "name": "Bash", "content": output}]
            report = verify_demo(client, workspace, runtime,
                                 f'exit_code=1\nFile "{workspace}/test_order.py"',
                                 test_hash, "done", events)
            saved_text = (workspace / "demo-report.json").read_text(encoding="utf-8")
            saved = json.loads(saved_text)
            self.assertNotIn(str(workspace), saved_text)
            self.assertEqual(saved["workspace"], "workspace/demo")
            self.assertTrue(saved["success"])
            self.assertEqual(report["workspace"], str(workspace))
            self.assertEqual(report["events"], events)


if __name__ == "__main__":
    unittest.main()
