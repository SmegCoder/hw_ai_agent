import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hw_ai_agent import cli
from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.budget import BudgetLimits
from hw_ai_agent.policy import Policy
from hw_ai_agent.session import SessionStore
from hw_ai_agent.skill_market import MarketError, SkillMarket
from hw_ai_agent.skills import SkillCatalog
from hw_ai_agent.tools import ToolError, ToolRuntime


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "skill-market"
WEEK = "study-week-overview"
NEXT = "next-study-task"


def call(identifier, tool, name):
    return {"tool_calls": [{"id": identifier, "type": "function",
                            "function": {"name": tool, "arguments": json.dumps({"name": name})}}]}


class SkillMarketTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.source = self.root / "market"
        shutil.copytree(FIXTURES, self.source)
        self.catalog = self.source / "catalog.json"
        self.config = self.source / "market.json"
        self.db_path = self.root / "agent" / "sessions.sqlite3"
        self.publish("v1")

    def publish(self, version):
        shutil.copyfile(self.source / f"catalog-{version}.json", self.catalog)

    def settings(self, **options):
        self.config.write_text(json.dumps({**json.loads(self.config.read_text()), **options}))

    def market(self):
        return SkillMarket(self.config, self.db_path, self.workspace)

    def runtime(self, *, approve=lambda _: True, policy=None):
        return ToolRuntime(self.workspace, skills=SkillCatalog(self.workspace), market=self.market(),
                           policy=policy, approve=approve)

    def events(self):
        return [json.loads(line) for line in self.market().events()]

    def test_agent_selects_installs_and_loads_only_relevant_offer(self):
        self.publish("v2")
        messages_seen = []

        class Model:
            def complete(self, messages, tools):
                messages_seen.append(messages[-1]["content"])
                installed = [msg for msg in messages if msg.get("name") == "install_skill"]
                loaded = [msg for msg in messages if msg.get("name") == "load_skill"]
                if loaded:
                    return {"content": "План включает буфер недели."}
                if installed:
                    return call("load", "load_skill", WEEK)
                self_context = messages[0]["content"]
                if WEEK in self_context and "план на неделю" in messages[-1]["content"]:
                    return call("install", "install_skill", WEEK)
                return {"content": "Не требуется скилл."}

        with SessionStore(self.db_path) as store:
            agent = CodingAgent(Model(), self.runtime(), store=store, max_rounds=4,
                                budget_limits=BudgetLimits(model_requests=4))
            self.assertEqual(agent.run("Составь учебный план на неделю"), "План включает буфер недели.")
            run = store.get(agent.session_id)
            self.assertIn("Буфер недели", messages_seen[-1])
            self.assertEqual([e["name"] for e in run.events if e["event"] == "skill_loaded"], [WEEK])
            self.assertEqual(store.budget(run.id)["spent"]["side_effects"], 1)
        self.assertEqual([row["name"] for row in self.market().installed()], [WEEK])
        self.assertEqual([(e["name"], e["run_id"]) for e in self.events() if e["event"] == "market_install"],
                         [(WEEK, run.id)])

    def test_startup_update_persists_and_old_run_has_pinned_revisions(self):
        initial = self.runtime()
        _, events = initial.prepare_skills()
        self.assertEqual([(e["event"], e.get("name")) for e in events],
                         [("market_check", None), ("market_new", WEEK)])
        initial.execute("install_skill", {"name": WEEK})
        old_document = initial.execute("load_skill", {"name": WEEK})
        old_run = self.runtime()
        old_run.prepare_skills()

        self.publish("v2")
        fresh = self.runtime()
        _, events = fresh.prepare_skills()
        update = next(e for e in events if e["event"] == "market_update")
        self.assertEqual((update["old_version"], update["new_version"], update["status"]),
                         ("1.0.0", "1.1.0", "updated"))
        self.assertNotEqual(update["old_sha256"], update["new_sha256"])
        self.assertEqual([e["name"] for e in events if e["event"] == "market_new"], [NEXT])
        self.assertIn("Буфер недели", fresh.execute("load_skill", {"name": WEEK}))
        restored = self.runtime()
        restored.restore_skills(old_run.skill_manifest, old_run.market_run, old_run.market_run_id)
        self.assertEqual(restored.execute("load_skill", {"name": WEEK}), old_document)
        initial_restored = self.runtime()
        initial_restored.restore_skills(initial.skill_manifest, initial.market_run, initial.market_run_id)
        self.assertEqual(initial_restored.execute("load_skill", {"name": WEEK}), old_document)
        initial_restored.execute("install_skill", {"name": WEEK})
        self.assertEqual([e["status"] for e in fresh.prepare_skills()[1]], ["unchanged"])
        self.assertEqual([r["version"] for r in self.market().installed()], ["1.1.0"])
        self.assertEqual(sum(e["event"] == "market_install" for e in self.events()), 1)

    def test_cli_resume_loads_installed_revision_after_another_run_updates_it(self):
        loaded_documents = {}

        class Model:
            def __init__(self, settings, **kwargs):
                self.settings = settings
                self.max_tokens = kwargs["max_tokens"]

            def complete(self, messages, tools):
                last = messages[-1]
                request = next(msg["content"] for msg in reversed(messages) if msg["role"] == "user")
                if last.get("name") == "load_skill":
                    loaded_documents[request] = last["content"]
                    return {"content": "План составлен."}
                if request == "Составь первый план" and last["role"] == "user":
                    return call("install", "install_skill", WEEK)
                return call("load", "load_skill", WEEK)

        policy = self.root / "policy.json"
        policy.write_text(json.dumps({"defaults": {"load_skill": "allow", "install_skill": "allow"}}))
        start = ["run", "--workspace", str(self.workspace), "--policy", str(policy),
                 "--market-config", str(self.config), "--provider", "ollama", "--model", "demo"]

        def invoke(args):
            argv = ["agi", *args, "--state-db", str(self.db_path), "--env-file", str(self.root / "missing.env")]
            with patch.object(sys, "argv", argv), patch.object(cli, "OpenAICompatibleClient", Model):
                return cli.main()

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(invoke([*start, "--stop-after-tools", "1", "Составь первый план"]), 130, err.getvalue())
            with SessionStore(self.db_path) as store:
                original_id = store.db.execute("SELECT id FROM sessions").fetchone()["id"]
            self.publish("v2")
            self.assertEqual(invoke([*start, "Составь второй план"]), 0, err.getvalue())
            self.assertEqual(invoke(["resume", original_id]), 0, err.getvalue())
        self.assertIn("next-study-task", out.getvalue())
        self.assertIn("Буфер недели", loaded_documents["Составь второй план"])
        self.assertNotIn("Буфер недели", loaded_documents["Составь первый план"])

    def test_toggles_bad_package_and_missing_source(self):
        runtime = self.runtime()
        runtime.prepare_skills()
        runtime.execute("install_skill", {"name": WEEK})
        self.publish("v2")
        self.settings(check_on_start=False, auto_update=True)
        self.assertEqual(runtime.prepare_skills()[1], [])
        self.assertEqual(self.market().installed()[0]["version"], "1.0.0")
        self.settings(check_on_start=True, auto_update=False)
        events = runtime.prepare_skills()[1]
        self.assertEqual(next(e["status"] for e in events if e["event"] == "market_update"), "available")
        self.assertEqual(self.market().installed()[0]["version"], "1.0.0")

        self.settings(auto_update=True)
        self.publish("invalid")
        events = runtime.prepare_skills()[1]
        self.assertEqual(next(e["status"] for e in events if e["event"] == "market_update"), "error")
        self.catalog.unlink()
        self.assertEqual(runtime.prepare_skills()[1][0]["status"], "error")
        self.assertIn("План учебной недели", runtime.execute("load_skill", {"name": WEEK}))
        self.assertEqual(self.market().installed()[0]["version"], "1.0.0")

    def test_policy_plan_and_offer_advertised_names(self):
        refused = self.runtime(approve=lambda _: False)
        refused.prepare_skills()
        with self.assertRaisesRegex(ToolError, "APPROVAL_REQUIRED"):
            refused.execute("install_skill", {"name": WEEK})
        denied = self.runtime(policy=Policy.from_dict({"defaults": {"install_skill": "deny"}}))
        denied.prepare_skills()
        self.assertNotIn("install_skill", [s["function"]["name"] for s in denied.tool_schemas])
        runtime = self.runtime()
        runtime.prepare_skills()
        with self.assertRaisesRegex(ToolError, "UNKNOWN_SKILL"):
            runtime.execute("install_skill", {"name": "deadline-report"})
        runtime.phase = "planning"
        with self.assertRaisesRegex(ToolError, "PLAN_MODE"):
            runtime.execute("install_skill", {"name": WEEK})
        self.assertEqual(self.market().installed(), [])

    def test_package_path_and_digest_enforced_and_run_offer_cannot_be_replaced(self):
        runtime = self.runtime()
        prompt, _ = runtime.prepare_skills()
        self.assertNotIn("packages/", prompt)
        package = self.source / runtime.market_offers[WEEK].path
        package.write_text(package.read_text() + "\nChanged on disk")
        with self.assertRaisesRegex(ToolError, "SHA-256"):
            runtime.execute("install_skill", {"name": WEEK})
        package.unlink()
        package.symlink_to(self.source / "packages" / WEEK / "1.1.0" / "SKILL.md")
        with self.assertRaisesRegex(ToolError, "Некорректный пакет"):
            runtime.execute("install_skill", {"name": WEEK})
        self.assertEqual(self.market().installed(), [])
        with self.assertRaisesRegex(MarketError, "вне рабочей папки"):
            SkillMarket(self.workspace / "market.json", self.db_path, self.workspace)
