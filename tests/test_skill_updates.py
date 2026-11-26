import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hw_ai_agent.config import Settings
from hw_ai_agent.conversation import ConversationShell
from hw_ai_agent.eval_cases import CASES
from hw_ai_agent.eval_report import summarize
from hw_ai_agent.eval_runner import EvalError, Options, run_lock, save_json
from hw_ai_agent.skill_market import MarketError, SkillMarket
from hw_ai_agent.skill_updates import decision_for, evaluate_candidate


NAME = "tracker-json-deadlines"
FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "skill-update"


def evaluation_rows():
    return [{"case_id": case.id, "variant": side, "trial": trial, "outcome_pass": True, "tools_pass": True,
             "routing_pass_or_na": True, "model_requests": 2, "input_tokens": 100, "output_tokens": 20,
             "input_usage_requests": 2, "output_usage_requests": 2,
             "skill_activations": {NAME: 1} if case.id == "deadlines" else {},
             "tool_calls": {}, "mcp_calls": {}, "error": None}
            for case in CASES for side in ("A", "B") for trial in (1, 2, 3)]


def summarize_comparison(rows):
    return summarize(rows, 30, {"cases": [{"id": case.id} for case in CASES]})


class SkillUpdateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.source = self.root / "registry"
        shutil.copytree(FIXTURES, self.source)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.output = self.root / "comparison"
        self.market = SkillMarket(self.source / "market.json", self.root / "store" / "sessions.sqlite3", self.workspace)
        self.settings = Settings("ollama", "test-model", "http://localhost:11434/v1", "ollama")
        self.publish("1.0.0")
        source, _ = self.market.check()
        _, offers = self.market.available(source, set())
        self.market.install(source, offers[NAME], "run-series")
        self.publish("1.1.0")

    def publish(self, version):
        shutil.copyfile(self.source / f"catalog-{version}.json", self.source / "catalog.json")

    def evaluate(self, rows, *, interrupt=False, while_running=None):
        def evaluator(output, settings, options, *, comparison, skill_versions):
            self.assertEqual(options.trials, 3)
            self.assertEqual(settings, self.settings)
            self.assertNotEqual(skill_versions["A"].document, skill_versions["B"].document)
            self.assertEqual(skill_versions["A"].name, skill_versions["B"].name)
            self.assertEqual(self.market.installed()[0]["version"], "1.0.0")
            output.mkdir(parents=True, exist_ok=True)
            save_json(output / "summary.json", summarize_comparison(rows))
            if while_running:
                while_running()
            return 130 if interrupt else 0

        with patch("hw_ai_agent.skill_updates.run_evaluation", evaluator):
            return evaluate_candidate(self.market, NAME, self.settings, Options(), output=self.output)

    def test_accept_then_rollback_through_chat_without_reapplying_same_version(self):
        _, events = self.market.check(startup=True)
        candidates = [e for e in events if e["event"] == "market_update"]
        self.assertEqual([e["status"] for e in candidates], ["pending"])
        self.assertEqual(self.market.installed()[0]["version"], "1.0.0")
        result, decision = self.evaluate(evaluation_rows())
        self.assertEqual(result, 0)
        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(self.market.installed()[0]["version"], "1.1.0")
        self.assertEqual(json.loads((self.output / "decision.json").read_text()), decision)

        agent = SimpleNamespace(runtime=SimpleNamespace(market=self.market), plan_mode=False)
        shell = ConversationShell(SimpleNamespace(store=None), agent, restore_runtime=None,
                                  report_answer=None, resume_command=None, approve_action=None, progress=None)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            shell.dispatch("/market-rollback", NAME, f"/market-rollback {NAME}")
        self.assertIn("восстановлена версия 1.0.0", stdout.getvalue())
        self.market.check(startup=True)
        self.assertEqual(self.market.installed()[0]["version"], "1.0.0")
        self.assertEqual([row["status"] for row in self.market.candidates()], ["rolled_back"])
        audit = [json.loads(line)["event"] for line in self.market.events()]
        self.assertIn("market_decision", audit)
        self.assertIn("market_rollback", audit)

    def test_reject_quality_regression_without_changing_active(self):
        self.market.check()
        rows = evaluation_rows()
        target = next(row for row in rows if row["variant"] == "B" and row["case_id"] == "deadlines")
        target["outcome_pass"] = False
        status, decision = self.evaluate(rows)
        self.assertEqual(status, 0)
        self.assertEqual(decision["status"], "rejected")
        self.assertIn("deadlines", decision["reasons"][0])
        self.market.check()
        self.assertEqual(self.market.installed()[0]["version"], "1.0.0")
        self.assertEqual([row["status"] for row in self.market.candidates()], ["rejected"])

    def test_incomplete_usage_or_policy_failure_blocks_acceptance(self):
        rows = evaluation_rows()
        candidate = next(row for row in rows if row["variant"] == "B")
        candidate.update(input_usage_requests=1, tools_pass=False)
        result = decision_for(summarize_comparison(rows), {"name": NAME})
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(len(result["reasons"]), 2)
        self.assertIsNone(result["token_sums"]["B"])

    def test_token_threshold_compares_sums_without_rounding(self):
        rows = evaluation_rows()
        for row in rows:
            if row["variant"] == "B":
                row["input_tokens"] += 24
        self.assertEqual(decision_for(summarize_comparison(rows), {"name": NAME})["status"], "accepted")
        next(row for row in rows if row["variant"] == "B")["input_tokens"] += 1
        self.assertEqual(decision_for(summarize_comparison(rows), {"name": NAME})["status"], "rejected")

    def test_interruption_keeps_candidate_pending_and_source_retraction_prevents_promotion(self):
        self.market.check()
        self.assertEqual(self.evaluate(evaluation_rows(), interrupt=True), (130, None))
        self.assertEqual([row["status"] for row in self.market.candidates()], ["pending"])
        with self.assertRaisesRegex(MarketError, "кандидат изменился"):
            self.evaluate(evaluation_rows(), while_running=lambda: self.publish("1.0.0"))
        self.assertEqual(self.market.installed()[0]["version"], "1.0.0")
        self.assertFalse((self.output / "decision.json").exists())

    def test_concurrent_comparison_cannot_start_in_a_second_output_directory(self):
        self.market.check()
        with run_lock(self.market.db_path.parent, f".{self.market.db_path.stem}-{NAME}.lock"):
            with patch("hw_ai_agent.skill_updates.run_evaluation") as evaluator:
                with self.assertRaisesRegex(EvalError, "уже выполняется"):
                    evaluate_candidate(self.market, NAME, self.settings, Options(), output=self.output)
                evaluator.assert_not_called()
