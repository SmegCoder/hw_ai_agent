import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hw_ai_agent import cli
from hw_ai_agent.anthropic_market import ANTHROPIC_REPOSITORY, Publication, PublicMarketError, published_names
from hw_ai_agent.skill_market import MarketError, SkillMarket
from hw_ai_agent.skills import Skill, SkillCatalog
from hw_ai_agent.tools import ToolRuntime


NAME = "brand-guidelines"


def publication(revision, body):
    card = Skill.parse(NAME, f"---\nname: {NAME}\ndescription: Brand styling guidelines\n---\n{body}\n".encode())
    return Publication(revision * 40, {NAME: card}, {})


class AnthropicMarketTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        config = root / "market.json"
        config.write_text(json.dumps({"marketplace": "anthropic", "check_on_start": False, "auto_update": True}))
        self.market = SkillMarket(config, root / "state" / "sessions.sqlite3", self.workspace)
        self.remote = self.enterContext(patch("hw_ai_agent.skill_market.read_publication"))
        self.remote.return_value = publication("a", "First set of brand colors.")

    def runtime(self, approve):
        return ToolRuntime(self.workspace, skills=SkillCatalog(self.workspace), market=self.market, approve=approve)

    def test_market_search_uses_anthropic_by_default_and_by_name(self):
        command = ["agi", "market", "search", "--workspace", str(self.workspace),
                   "--state-db", str(self.market.db_path.parent / "sessions.sqlite3")]
        for options in ([], ["--market-config", "anthropic"]):
            with self.subTest(options=options):
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(sys, "argv", command + options), redirect_stdout(stdout), redirect_stderr(stderr):
                    self.assertEqual(cli.main(), 0, stderr.getvalue())
                self.assertIn(NAME, stdout.getvalue())
        self.assertEqual(self.remote.call_count, 2)

    def test_missing_market_file_and_invalid_json_have_distinct_errors(self):
        catalog = self.workspace.parent / "broken.json"
        market = SkillMarket(catalog, self.market.db_path, self.workspace)
        with self.assertRaisesRegex(MarketError, "Файл JSON не найден · broken.json"):
            market.config()
        catalog.write_text('{"marketplace":\n}', encoding="utf-8")
        with self.assertRaisesRegex(MarketError, "Некорректный JSON · broken.json, строка 2, столбец 1"):
            market.config()
        self.remote.assert_not_called()

    def test_declared_card_paths_cannot_leave_repository_skill_directory(self):
        def manifest(path):
            return json.dumps({"name": "anthropic-agent-skills",
                               "plugins": [{"source": "./", "skills": [path]}]}).encode()

        self.assertEqual(published_names(manifest("./skills/brand-guidelines")), [NAME])
        for path in ("../skills/brand-guidelines", "./skills/../brand-guidelines", "https://example.org/skill"):
            with self.subTest(path=path), self.assertRaises(PublicMarketError):
                published_names(manifest(path))

    def test_install_approved_card_from_cache_after_git_failure(self):
        source, events = self.market.check()
        self.assertEqual(source, ANTHROPIC_REPOSITORY)
        self.assertEqual(events[0]["revision"], "a" * 40)
        offer, = self.market.search(source, "BRAND styling")
        self.assertEqual(offer.name, NAME)
        self.assertEqual(self.market.installed(), [])

        self.remote.side_effect = PublicMarketError("Git unavailable")
        self.assertEqual(self.market.check()[1][0]["status"], "error")
        self.remote.reset_mock()
        approvals = []
        runtime = self.runtime(lambda request: approvals.append(request) or True)
        prompt, _ = runtime.prepare_skills()
        self.assertIn(NAME, prompt)
        runtime.execute("install_skill", {"name": NAME})
        self.assertIn(ANTHROPIC_REPOSITORY, approvals[0].preview)
        self.assertIn(offer.sha256, approvals[0].preview)
        self.assertIn("First set of brand colors", runtime.execute("load_skill", {"name": NAME}))
        self.assertEqual(self.market.installed()[0]["source"], ANTHROPIC_REPOSITORY)
        self.remote.assert_not_called()

    def test_public_update_is_staged_only_when_card_changes(self):
        source, _ = self.market.check()
        old_offer, = self.market.search(source, NAME)
        self.market.install(source, old_offer, "initial-run")

        self.remote.return_value = publication("b", "First set of brand colors.")
        self.assertEqual(self.market.check()[1][0]["status"], "unchanged")
        self.assertEqual(self.market.candidates(), [])

        new_publication = publication("c", "Second set of brand colors.")
        self.remote.return_value = new_publication
        self.market.check()
        candidate, old, new = self.market.candidate(NAME)
        self.assertEqual((candidate["status"], old.digest, new.digest),
                         ("pending", old_offer.sha256, new_publication.cards[NAME].digest))
        self.assertEqual(self.market.installed()[0]["sha256"], old.digest)

        decision = {"status": "accepted", "reasons": [], "report": "test"}
        self.remote.return_value = publication("d", "Third set of brand colors.")
        with self.assertRaises(MarketError):
            self.market.finish_candidate(candidate["id"], decision)
        self.assertEqual(self.market.installed()[0]["sha256"], old.digest)

        self.remote.return_value = new_publication
        self.market.finish_candidate(candidate["id"], decision)
        self.assertEqual(self.market.installed()[0]["sha256"], new.digest)
        self.market.rollback(NAME)
        self.assertEqual(self.market.installed()[0]["sha256"], old.digest)
