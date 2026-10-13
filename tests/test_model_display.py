from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hw_ai_agent import cli
from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.config import Settings
from hw_ai_agent.model_catalog import Model, comparison
from hw_ai_agent.pricing import Price
from hw_ai_agent.routing import shortlist, pick_model
from hw_ai_agent.session import SessionStore
from hw_ai_agent.tools import ToolRuntime


class DisplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = self.root / '.env'
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start(); self.addCleanup(self.environment.stop)

    def invoke(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with patch('sys.argv', ['hw-ai-agent', *map(str, arguments)]), redirect_stdout(out), redirect_stderr(err):
            result = cli.main()
        return result, out.getvalue(), err.getvalue()

    def catalog(self):
        models = [Model('a/paid', Price('0.000001', '0.000002'), 64000, None, True)]
        models += [Model(f'b/free-{i:02}', Price('0', '0'), 64000, None, True) for i in range(70)]
        models += [Model('c/paid', Price('0.000002', '0.000003'), 64000, None, True), Model('d/unknown', None, 64000, None, True)]
        catalog = SimpleNamespace(provider='openrouter', source='fixture', updated_at='fixture', status='fixture', warning=None, models=models)
        catalog.find = lambda name: next((m for m in models if m.id==name), None)
        return catalog

    def models(self, *arguments):
        with patch('hw_ai_agent.cli.catalog_for', return_value=self.catalog()):
            result, out, err = self.invoke('models', '--provider', 'openrouter', '--env-file', self.env,
                                         '--state-db', self.root/'state', '--json', *arguments)
        self.assertEqual(result, 0, err)
        return json.loads(out)

    def test_default_list_is_50_by_name_and_includes_paid_models(self):
        report = self.models()
        self.assertEqual((report['shown'], report['matched'], report['sort']), (50, 73, 'name'))
        self.assertEqual(report['models'][0]['id'], 'a/paid')
        self.assertGreater(Decimal(report['models'][0]['scenario_usd']), 0)

    def test_all_includes_entire_catalog_and_overrides_limit(self):
        report = self.models('--all', '--limit', 2)
        self.assertEqual(report['shown'], 73)
        self.assertEqual(self.models('--limit', 120)['shown'], 73)

    def test_paid_free_filters_exclude_unknown_prices(self):
        paid = self.models('--paid', '--all', '--input-tokens', 0, '--output-tokens', 0)
        self.assertEqual([m['id'] for m in paid['models']], ['a/paid', 'c/paid'])
        free = self.models('--free', '--all')
        self.assertEqual(free['shown'], 70)
        self.assertTrue(all(m['id'].startswith('b/free-') for m in free['models']))

    def test_price_sort_is_opt_in_and_search_composes_with_filters(self):
        report = self.models('--sort', 'price', '--limit', 3)
        self.assertTrue(all(m['scenario_usd']=='0' for m in report['models']))
        report = self.models('--search', 'c/', '--paid', '--all')
        self.assertEqual([m['id'] for m in report['models']], ['c/paid'])

    def test_price_filters_are_mutually_exclusive(self):
        with patch('hw_ai_agent.cli.catalog_for') as catalog:
            with self.assertRaises(SystemExit) as caught:
                self.invoke('models', '--paid', '--free')
            self.assertEqual(caught.exception.code, 2)
            catalog.assert_not_called()

    def test_table_columns_are_aligned_for_long_ids_and_unknown_numbers(self):
        models = [Model('x/short', Price('0.000001', '0.000002'), 1000, None, True),
                  Model('vendor/a-much-longer-model-id', None, 1000000, None, False)]
        table = comparison(models, numbered=True)
        lines = [line for line in table.splitlines() if ' | ' in line]
        positions = [[i for i, char in enumerate(line) if char=='|'] for line in lines]
        self.assertTrue(all(p==positions[0] for p in positions))
        self.assertIn('1 000 000', table)
        self.assertIn('vendor/a-much-longer-model-id', table)
        self.assertIn('неизвестно', table)

    def test_picker_is_20_choices_with_numbers_in_table(self):
        catalog = self.catalog()
        self.assertEqual(len(shortlist(catalog)), 20)
        out = []
        selected = pick_model(catalog, read=lambda prompt: '20', show=out.append)
        self.assertEqual(selected.id, shortlist(catalog)[19].id)
        self.assertTrue(any('№' in text and ' | ' in text for text in out))
        self.assertFalse(any(' — ' in text for text in out))

    def test_sdk_estimate_off_skips_calculation_but_keeps_billing(self):
        workspace = self.root/'workspace'; workspace.mkdir()
        class Client:
            settings = Settings('openrouter', 'fixture', 'https://openrouter.ai/api/v1', 'key')
            price = Price('0.000001', '0.000002')
            max_tokens = 100
            def __init__(self): self.usage = []
            def complete(self, messages, tools):
                self.usage.append({'cost': '0.003'})
                return {'content': 'done'}
        with SessionStore(self.root/'state') as store, patch('hw_ai_agent.agent.estimate', side_effect=AssertionError('Estimate disabled')):
            agent = CodingAgent(Client(), ToolRuntime(workspace), store=store, cost_estimate=False)
            agent.run('Task')
            self.assertFalse(any(e['event']=='model_cost_estimate' for e in agent.events))
            self.assertEqual(store.cost(agent.session_id)['reported_usd'], '0.003')
            self.assertEqual(store.budget(agent.session_id)['spent']['model_requests'], 1)

    def test_env_default_and_cli_override_apply_to_run(self):
        workspace = self.root/'workspace'; workspace.mkdir()
        class Client:
            def __init__(self, settings, **kwargs):
                self.settings, self.max_tokens, self.usage = settings, kwargs['max_tokens'], []
            def complete(self, messages, tools): return {'content': 'done'}
        self.env.write_text('AGENT_PROVIDER=ollama\nAGENT_COST_ESTIMATE=false\n')
        for flags, expected in (([], False), (['--cost-estimate'], True), (['--no-cost-estimate'], False)):
            with patch('hw_ai_agent.cli.OpenAICompatibleClient', side_effect=Client):
                status, out, err = self.invoke('run', '--env-file', self.env, '--workspace', workspace, '--state-db', self.root/'state', *flags, 'Task')
            self.assertEqual(status, 0, err)
            self.assertEqual('Оценка запроса:' in out, expected)
            self.assertIn('Оплата API', out)

    def test_set_persists_estimate_toggle(self):
        self.assertEqual(self.invoke('set', '--env-file', self.env, '--no-cost-estimate')[0], 0)
        self.assertIn('AGENT_COST_ESTIMATE="false"', self.env.read_text())
        self.assertEqual(self.invoke('set', '--env-file', self.env, '--cost-estimate')[0], 0)
        self.assertIn('AGENT_COST_ESTIMATE="true"', self.env.read_text())

    def test_resume_toggle_changes_display_without_changing_saved_limits(self):
        workspace, db = self.root/'demo', self.root/'state'
        self.env.write_text('AGENT_COST_ESTIMATE=false\n')
        status, out, err = self.invoke('plan-demo', '--offline', '--env-file', self.env, '--workspace', workspace,
            '--state-db', db, '--stop-after-tools', 1, '--max-rounds', 17, '--max-tokens', 1400)
        self.assertEqual(status, 130, err)
        self.assertNotIn('Оценка запроса:', out)
        with SessionStore(db) as store:
            identifier = store.list_sessions()[0]['id']
            before = store.budget(identifier)['limits']
        # Invalid NEW limits must not be applied to the saved session.
        self.env.write_text('AGENT_COST_ESTIMATE=false\nAGENT_MAX_ROUNDS=bad\n')
        os.environ.clear()
        with patch('hw_ai_agent.cli.confirm_plan', return_value=False):
            status, out, err = self.invoke('resume', identifier, '--state-db', db, '--env-file', self.env, '--cost-estimate')
        self.assertEqual(status, 1, err)
        self.assertIn('Оценка запроса:', out)
        with SessionStore(db) as store:
            self.assertEqual(store.budget(identifier)['limits'], before)
            self.assertEqual(store.get(identifier).metadata['max_tokens'], 1400)
            self.assertEqual(store.cost(identifier)['known_usd'], '0')


if __name__ == '__main__': unittest.main()
