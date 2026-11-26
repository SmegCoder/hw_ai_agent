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
from hw_ai_agent.skills import Skill


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
        comparison = config.get("comparison")
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
        if (variant == "B" or comparison) and case.skills:
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
                test.assertEqual("load_skill" in names, bool(variant == "B" or comparison))
                self.usage.append({"prompt_tokens": 10} if len(replies) == 1 else
                                  {"prompt_tokens": 10, "completion_tokens": 5})
                return replies.pop(0)

        with patch.object(eval_runner, "OpenAICompatibleClient", Model):
            if comparison:
                eval_runner.execute_trial(directory, case, variant, settings, options,
                                          skill_dir=eval_runner.version_directory(manifest.parent, variant))
            else:
                eval_runner.execute_trial(directory, case, variant, settings, options)
        if len(self.started) == 1 and self.fail_first == "interrupt":
            return "interrupted"
        return None

    def rows(self):
        return [json.loads(line) for line in (self.output / "results.jsonl").read_text().splitlines()]

    def test_comparison_uses_same_tools_and_different_versions_in_both_arms(self):
        name = "tracker-test"
        versions = {side: Skill.parse(name, f"---\nname: {name}\ndescription: Export tasks\n---\n{side}\n".encode())
                    for side in ("A", "B")}
        comparison = {"name": name, "focus": ["deadlines"], "revisions": {
            side: {"version": side, "sha256": skill.digest} for side, skill in versions.items()}}
        settings = eval_runner.Settings("ollama", "test-model", "http://localhost:11434/v1", "ollama")
        stdout = io.StringIO()
        with (redirect_stdout(stdout), patch.dict(os.environ, self.environment, clear=True),
              patch.object(eval_runner, "launch_trial", self.local_launch)):
            self.assertEqual(eval_runner.run_evaluation(self.output, settings, eval_runner.Options(),
                             comparison=comparison, skill_versions=versions), 0)
        rows = self.rows()
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(row["outcome_pass"] and row["tools_pass"] and row["routing_pass_or_na"] is True
                            for row in rows))
        for row in (row for row in rows if row["case_id"] == "deadlines"):
            journal = self.output / row["events"]
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            [loaded] = [event for event in events if event["type"] == "tool_result"
                        and event["data"]["name"] == "load_skill"]
            self.assertEqual(loaded["data"]["content"], versions[row["variant"]].document)

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
        self.assertIn(f"[run-001]: {first['events']} \"{first['run_id']}\"", report)
        self.assertIn("частично", report)
        self.assertIn("MCP_TOOL_ERROR", report)
        self.assertTrue(all((self.output / row["events"]).is_file() for row in rows))
        self.assertEqual(self.cli_run()[0], 0)
        self.assertEqual(len(self.started), 30)
        results = (self.output / "results.jsonl").read_bytes()
        (self.output / "summary.md").write_text("Старый отчёт\n", encoding="utf-8")
        with (patch.dict(os.environ, {}, clear=True),
              patch.object(sys, "argv", ["agi", "eval", "report", "--output", str(self.output)]),
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            self.assertEqual(cli.main(), 0)
        self.assertEqual((self.output / "results.jsonl").read_bytes(), results)
        self.assertEqual((self.output / "summary.md").read_text(), report)
        for block in report.split("\n\n"):
            if block.startswith("|"):
                positions = lambda line: [i for i, char in enumerate(line) if char == "|"]
                header, *body = block.splitlines()
                self.assertTrue(all(positions(line) == positions(header) for line in body))

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
