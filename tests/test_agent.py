from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hw_ai_agent.agent import AgentError, CodingAgent
from hw_ai_agent.tools import ToolRuntime


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ScriptedClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[list[Mapping[str, Any]]] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        self.requests.append(list(messages))
        if not self.responses:
            raise AssertionError("the agent requested an unexpected model response")
        return self.responses.pop(0)


class CodingAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.runtime = ToolRuntime(self.root, approve=lambda request: True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_multiple_tool_calls_are_executed_in_order(self) -> None:
        client = ScriptedClient(
            [
                {
                    "content": None,
                    "tool_calls": [
                        tool_call("one", "write_file", {"path": "answer.txt", "content": "42"}),
                        tool_call("two", "read_file", {"path": "answer.txt"}),
                    ],
                },
                {"content": "Finished."},
            ]
        )

        result = CodingAgent(client, self.runtime).run("Create and check answer.txt")

        self.assertEqual(result, "Finished.")
        self.assertEqual((self.root / "answer.txt").read_text(), "42")
        tool_messages = [message for message in client.requests[1] if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in tool_messages], ["one", "two"])
        self.assertEqual(tool_messages[1]["content"], "42")

    def test_tool_error_is_returned_to_model(self) -> None:
        client = ScriptedClient(
            [
                {"tool_calls": [tool_call("bad", "read_file", {"path": "../secret"})]},
                {"content": "I could not read outside the workspace."},
            ]
        )

        CodingAgent(client, self.runtime).run("Read the secret")

        tool_message = [message for message in client.requests[1] if message["role"] == "tool"][0]
        self.assertIn("ERROR:", tool_message["content"])
        self.assertIn("escapes the workspace", tool_message["content"])

    def test_invalid_json_is_returned_as_tool_error(self) -> None:
        client = ScriptedClient(
            [
                {
                    "tool_calls": [
                        {"id": "bad-json", "function": {"name": "read_file", "arguments": "{"}}
                    ]
                },
                {"content": "Recovered."},
            ]
        )

        self.assertEqual(CodingAgent(client, self.runtime).run("Continue after an error"), "Recovered.")
        tool_message = [message for message in client.requests[1] if message["role"] == "tool"][0]
        self.assertTrue(tool_message["content"].startswith("ERROR:"))

    def test_round_limit_stops_infinite_tool_loop(self) -> None:
        client = ScriptedClient(
            [{"tool_calls": [tool_call(str(index), "list_files", {})]} for index in range(2)]
        )

        with self.assertRaisesRegex(AgentError, "exceeded"):
            CodingAgent(client, self.runtime, max_rounds=2).run("Never finish")

    def test_test_write_is_rejected_and_agent_can_continue(self):
        test = self.root / "test_order.py"
        test.write_text("original", encoding="utf-8")
        client = ScriptedClient([
            {"tool_calls": [tool_call("blocked", "write_file", {"path": "test_order.py", "content": "changed"})]},
            {"tool_calls": [tool_call("fix", "write_file", {"path": "order.py", "content": "fixed"})]},
            {"content": "Fixed source, preserved tests."},
        ])
        self.assertEqual(CodingAgent(client, self.runtime).run("Fix source"), "Fixed source, preserved tests.")
        self.assertEqual(test.read_text(), "original")
        self.assertIn("read-only", client.requests[1][-1]["content"])
        self.assertEqual((self.root / "order.py").read_text(), "fixed")

    def test_local_model_arguments_and_missing_ids_are_normalized(self):
        client = ScriptedClient([
            {"tool_calls": [{"function": {"name": "list_files", "arguments": {"path": "."}}}]},
            {"content": "OK"},
        ])
        self.assertEqual(CodingAgent(client, self.runtime).run("List files"), "OK")
        assistant, result = client.requests[1][-2:]
        self.assertEqual(assistant["tool_calls"][0]["id"], result["tool_call_id"])
        self.assertIsInstance(assistant["tool_calls"][0]["function"]["arguments"], str)

    def test_empty_model_answer_and_duplicate_ids_are_rejected(self):
        bad = [{"content": None}, {"tool_calls": [tool_call("same", "list_files", {}), tool_call("same", "list_files", {})]}]
        for response in bad:
            with self.subTest(response=response), self.assertRaises(AgentError):
                CodingAgent(ScriptedClient([response]), self.runtime).run("Task")

    @unittest.skipUnless(os.getenv("REQUIRE_SANDBOX_TESTS") == "1", "Requires real OS sandbox")
    def test_agent_can_fix_order_module_and_run_tests_without_changing_test(self) -> None:
        test_source = (
            "import unittest\n"
            "from order import total\n\n"
            "class OrderTests(unittest.TestCase):\n"
            "    def test_total(self):\n"
            "        self.assertEqual(total([2, 3, 5]), 10)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        (self.root / "order.py").write_text("def total(values):\n    return 0\n", encoding="utf-8")
        (self.root / "test_order.py").write_text(test_source, encoding="utf-8")
        client = ScriptedClient(
            [
                {
                    "tool_calls": [
                        tool_call(
                            "fix",
                            "write_file",
                            {"path": "order.py", "content": "def total(values):\n    return sum(values)\n"},
                        ),
                        tool_call("test", "bash", {"command": "python -m unittest test_order.py -v"}),
                    ]
                },
                {"content": "Fixed order.py; tests pass."},
            ]
        )

        result = CodingAgent(client, self.runtime).run("Fix order.py without editing the test")

        self.assertIn("tests pass", result)
        self.assertEqual((self.root / "test_order.py").read_text(encoding="utf-8"), test_source)
        tool_messages = [message for message in client.requests[1] if message["role"] == "tool"]
        self.assertIn("exit_code=0", tool_messages[1]["content"])


if __name__ == "__main__":
    unittest.main()

