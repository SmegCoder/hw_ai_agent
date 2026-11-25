from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_fixtures import tool_call
from hw_ai_agent import cli, eval_runner
from hw_ai_agent.eval_cases import NOTE, OVERDUE, REFERENCE_DATE, TRACKER_TOOL, UNAVAILABLE
from hw_ai_agent.session import SessionStore


class EvalRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "series"
        self.environment = {"OPENROUTER_API_KEY": "test-eval-secret"}
        self.started = []
        self.snapshots = {}
        self.fail_first = None

    def cli_run(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        args = ["agi", "eval", "run", "--provider", "openrouter", "--model", "test/model",
                "--env-file", str(self.output / "unused.env"), "--output", str(self.output), *extra]
        with (patch.dict(os.environ, self.environment, clear=True),
              patch.object(sys, "argv", args), patch.object(eval_runner, "launch_trial", self.local_launch),
              redirect_stdout(stdout), redirect_stderr(stderr)):
            code = cli.main()
        return code, stdout.getvalue(), stderr.getvalue()

    def local_launch(self, manifest, directory, case, variant, options, lock_fd):
        self.assertTrue(manifest.is_file())
        os.fstat(lock_fd)
        self.assertEqual(list((directory / "workspace").iterdir()), [])
        snapshot = directory / "source" / "tasks.json"
        self.snapshots[case.id, variant] = snapshot.read_bytes() if snapshot.is_file() else None
        self.started.append((case.id, variant))
        config = json.loads(manifest.read_text())
        settings = eval_runner.Settings.from_env(config["provider"], config["model"])
        test = self
        if len(self.started) == 1 and self.fail_first == "timeout":
            # Simulate termination after the real runtime created its session.
            class Model:
                def __init__(self, settings, **kwargs):
                    self.settings = settings

                def complete(self, messages, tools):
                    raise KeyboardInterrupt

            with patch.object(eval_runner, "OpenAICompatibleClient", Model), self.assertRaises(KeyboardInterrupt):
                eval_runner.execute_trial(directory, case, variant, settings, options)
            with SessionStore(directory / "session.sqlite3") as store:
                run_id = store.list_sessions()[0]["id"]
                store.set_status(run_id, "running")
            return "timeout"

        replies = []
        if variant == "B" and case.skills:
            replies.append({"tool_calls": [tool_call("skill", "load_skill", {"name": case.skills[0]})]})
        if case.mcp != "unused":
            replies.append({"tool_calls": [tool_call("tracker", TRACKER_TOOL, {})]})
        if case.mcp == "error":
            replies.append({"content": UNAVAILABLE})
        else:
            if case.id == "note":
                content = NOTE
            elif case.id == "open-tasks":
                content = json.dumps(["T1", "T3", "T4", "T5"])
            elif case.id == "deadlines":
                content = json.dumps({"as_of": REFERENCE_DATE, "overdue": OVERDUE,
                                      "missing_due_date": 0, "skipped_records": 0})
            else:
                content = json.dumps({"overdue": [{key: t[key] for key in ("id", "due_date")} for t in OVERDUE],
                                      "missing_due_date_ids": ["T6", "T7"]})
            replies.extend(({"tool_calls": [tool_call("write", "write_file", {"path": case.output, "content": content})]},
                            {"content": "Записано."}))

        class Model:
            def __init__(self, settings, **kwargs):
                self.settings = settings
                self.usage = []

            def complete(self, messages, tools):
                names = {schema["function"]["name"] for schema in tools}
                test.assertIn(TRACKER_TOOL, names)
                test.assertEqual("load_skill" in names, variant == "B")
                self.usage.append({"prompt_tokens": 10} if len(replies) == 1 else
                                  {"prompt_tokens": 10, "completion_tokens": 5})
                return replies.pop(0)

        with patch.object(eval_runner, "OpenAICompatibleClient", Model):
            eval_runner.execute_trial(directory, case, variant, settings, options)
        if len(self.started) == 1 and self.fail_first == "interrupt":
            return "interrupted"
        return None

    def rows(self):
        return [json.loads(line) for line in (self.output / "results.jsonl").read_text().splitlines()]

    def test_thirty_attempts_resume_without_duplicate_and_include_evidence(self):
        self.fail_first = "interrupt"
        code, _, stderr = self.cli_run()
        self.assertEqual(code, 130, stderr)
        first = self.rows()[0]
        first_path = self.output / first["events"]
        original = first_path.read_bytes()
        code, out, stderr = self.cli_run()
        self.assertEqual(code, 0, stderr)
        self.assertIn("Сохранено попыток · 30/30", out)
        self.assertEqual(len(self.started), 30)
        rows = self.rows()
        self.assertEqual(first_path.read_bytes(), original)
        self.assertEqual(len({r["run_id"] for r in rows}), 30)
        self.assertEqual(Counter((r["case_id"], r["variant"]) for r in rows), Counter(self.started))
        self.assertTrue(all(n == 3 for n in Counter(self.started).values()))
        failures = [(r["case_id"], r["variant"], r["check"], r["error"])
                    for r in rows if not r["outcome_pass"] or not r["tools_pass"]]
        self.assertEqual(failures, [])
        self.assertTrue(all(r["routing_pass_or_na"] is True for r in rows if r["variant"] == "B"))
        self.assertTrue(all(r["routing_pass_or_na"] == "n/a" for r in rows if r["variant"] == "A"))
        for case_id, _ in self.started:
            self.assertEqual(self.snapshots[case_id, "A"], self.snapshots[case_id, "B"])
        self.assertEqual((self.output / "manifest.json").read_text().count("test-eval-secret"), 0)
        report = (self.output / "summary.md").read_text()
        self.assertIn(f"[{first['run_id']}]({first['events']})", report)
        self.assertIn("частично", report)
        self.assertIn("MCP_TOOL_ERROR", report)
        self.assertTrue(all((self.output / row["events"]).is_file() for row in rows))
        self.assertEqual(self.cli_run()[0], 0)
        self.assertEqual(len(self.started), 30)

    def test_timeout_retained_when_trial_row_must_be_rebuilt(self):
        self.fail_first = "timeout"
        code, _, stderr = self.cli_run("--trials", "1")
        self.assertEqual(code, 0, stderr)
        row = self.rows()[0]
        self.assertEqual(row["status"], "timeout")
        self.assertFalse(row["outcome_pass"])
        self.assertIsNone(row["input_tokens"])
        (self.output / row["events"]).with_name("result.json").unlink()
        self.assertEqual(self.cli_run("--trials", "1")[0], 0)
        self.assertEqual(len(self.started), 10)
        self.assertEqual(self.rows()[0], row)
        self.assertEqual(self.cli_run("--trials", "2")[0], 1)
        self.assertEqual(len(self.started), 10)


if __name__ == "__main__":
    unittest.main()
