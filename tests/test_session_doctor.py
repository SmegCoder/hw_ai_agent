from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hw_ai_agent import cli
from hw_ai_agent.budget import TRUNCATED
from hw_ai_agent.conversation import Conversation
from hw_ai_agent.session import SessionStore
from hw_ai_agent.session_doctor import inspect_session


class SessionDoctorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.path = root / "sessions.sqlite3"
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.store = SessionStore(self.path)
        self.addCleanup(self.store.db.close)
        self.call_number = 0

    def new_run(self, conversation_id=None):
        metadata = {"workspace": str(self.workspace), "provider": "ollama", "model": "test-model",
                    "journal_mode": conversation_id is not None,
                    "kind": "chat-turn" if conversation_id is not None else "run"}
        return self.store.create(metadata, [{"role": "user", "content": "Прочитай данные"}], 8, 10_000,
                                 conversation_id=conversation_id)

    def add_tool(self, run, name, arguments, result, *, failed=False):
        self.call_number += 1
        key = f"{self.call_number}:0"
        tags = {"call": key, "tool_call_id": f"mock-{key}", "name": name, "phase": "execution"}
        self.store.event(run, {"event": "tool_call", "arguments": arguments, **tags})
        self.store.event(run, {"event": "tool_result", "failed": failed, "content": result, **tags})
        return [row["id"] for row in list(self.store.event_records(run))[-2:]]

    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["agi", "doctor", "session", *map(str, arguments)]
        with (patch.object(sys, "argv", argv), patch.dict(os.environ, {}, clear=True),
              patch.object(cli.Settings, "from_env") as settings,
              patch.object(cli, "OpenAICompatibleClient") as model, redirect_stdout(stdout), redirect_stderr(stderr)):
            status = cli.main()
        model.assert_not_called()
        settings.assert_not_called()
        return status, stdout.getvalue(), stderr.getvalue()

    def test_cli_reports_repeats_with_real_event_ids_and_partial_api_usage(self):
        run = self.new_run()
        for _ in range(2):
            self.store.start_model(run)
        usage = {"prompt_tokens": 41, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 8}}
        for event in ("model_cost", "model_usage"):
            self.store.event(run, {"event": event, "round": 1, "usage": usage})
        self.store.event(run, {"event": "skills_discovered", "names": ["deadline-report"]})
        self.store.event(run, {"event": "skill_loaded", "name": "deadline-report"})
        identical_reads = self.add_tool(run, "list_files", '{"path":".","recursive":false}', "tasks.json")
        identical_reads += self.add_tool(run, "list_files", {"recursive": False, "path": "."}, "tasks.json")
        self.add_tool(run, "mcp__tracker__list_tasks", "{}", '{"tasks":[]}')
        failed = self.add_tool(run, "Bash", '{"command":"pwd"}', "ERROR: POLICY_DENIED: Bash", failed=True)
        failed += self.add_tool(run, "Bash", '{"command":"ls"}', "ERROR: POLICY_DENIED: Bash", failed=True)
        self.store.complete(run, "Готово.")
        original = self.path.read_bytes()
        status, text, error = self.invoke(run, "--state-db", self.path, "--json")
        self.assertEqual(status, 0, error)
        report = json.loads(text)
        self.assertEqual(report["session_id"], run)
        state = report["runs"][0]
        self.assertEqual(state["model_requests"], 2)
        self.assertEqual(state["usage"]["total_tokens"], {"tokens": 50, "reported_requests": 1, "requests": 2})
        self.assertEqual(state["usage"]["cached_tokens"]["tokens"], 8)
        self.assertIsNone(state["usage"]["cache_write_tokens"]["tokens"])
        self.assertEqual(state["skill_loads"], {"deadline-report": 1})
        self.assertEqual(state["mcp_calls"], {"tracker/list_tasks": 1})
        by_rule = {finding["kind"]: finding for finding in state["findings"]}
        self.assertEqual(by_rule["repeated_tool_error"]["event_ids"], failed)
        self.assertEqual(by_rule["repeated_tool_error"]["error_code"], "POLICY_DENIED")
        self.assertEqual(by_rule["repeated_tool_error"]["known_argument_sets"], 2)
        self.assertEqual(by_rule["repeated_unchanged_read"]["event_ids"], identical_reads)
        for finding in by_rule.values():
            self.assertEqual(finding["run_id"], run)
            self.assertEqual(finding["count"], 2)
            self.assertTrue(finding["recommendation"])
        self.assertEqual(report["doctor_usage"]["model_requests"], 0)
        status, text, error = self.invoke(run, "--state-db", self.path)
        self.assertEqual(status, 0, error)
        self.assertIn("41 (частично, 1/2)", text)
        self.assertIn("tracker/list_tasks × 1", text)
        self.assertIn("Bash / POLICY_DENIED", text)
        self.assertEqual(self.path.read_bytes(), original)

    def test_records_after_changes_or_in_different_runs_do_not_form_repeats(self):
        chat = Conversation.create(self.store, {})
        a, b = self.new_run(chat.id), self.new_run(chat.id)
        args = {"path": "tasks.json"}
        self.add_tool(a, "read_file", args, "baseline")
        self.add_tool(a, "write_file", {"path": "tasks.json", "content": "baseline"}, "wrote tasks.json")
        self.add_tool(a, "read_file", args, "baseline")
        self.add_tool(a, "read_file", args, "changed contents")
        self.add_tool(a, "read_file", {"path": "other.json"}, "changed contents")
        self.add_tool(a, "read_file", args, "a partial read" + TRUNCATED)
        self.add_tool(a, "read_file", args, "a partial read" + TRUNCATED)
        self.add_tool(a, "Bash", {"command": "pwd"}, "ERROR: POLICY_DENIED: Bash", failed=True)
        self.add_tool(a, "Bash", {"command": "ls"}, "ERROR: timeout", failed=True)
        self.add_tool(b, "read_file", args, "changed contents")
        self.add_tool(b, "Bash", {"command": "pwd"}, "ERROR: POLICY_DENIED: Bash", failed=True)
        self.store.event(b, {"event": "tool_call", "name": "read_file", "arguments": args, "call": "99:0"})
        self.store.start_model(b)
        self.store.complete(a, "Готово.")
        self.store.complete(b, "Готово.")
        chat.end()

        report = inspect_session(self.path, chat.id)
        self.assertEqual([state["run_id"] for state in report["runs"]], [a, b])
        self.assertEqual(report["total"]["tool_calls"], {"read_file": 8, "write_file": 1, "Bash": 3})
        self.assertEqual(report["total"]["model_requests"], 1)
        self.assertIsNone(report["total"]["usage"]["prompt_tokens"]["tokens"])
        self.assertEqual([state["findings"] for state in report["runs"]], [[], []])
        self.assertTrue(any("обрезанн" in limit.lower() for limit in report["runs"][0]["limitations"]))
        self.assertTrue(any("сопоставить" in limit for limit in report["runs"][1]["limitations"]))

    def test_missing_database_is_not_created(self):
        path = self.path.with_name("missing.sqlite3")
        status, _, error = self.invoke("a" * 32, "--state-db", path)
        self.assertEqual(status, 1)
        self.assertIn("для чтения", error)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
