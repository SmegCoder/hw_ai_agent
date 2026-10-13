from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hw_ai_agent.cli import main, choose_settings
from hw_ai_agent.config import Settings
from hw_ai_agent.model_catalog import Model
from hw_ai_agent.pricing import Price, PricingError
from hw_ai_agent.routing import complexity, choose_route, pick_model, shortlist
from hw_ai_agent.session import SessionStore
from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.tools import ToolRuntime


def fixture_catalog():
    cheap = Price('0.0000001', '0.0000002', source='fixture')
    regular = Price('0.000001', '0.000002', source='fixture')
    models = [Model('deepseek/deepseek-v4.1-flash', cheap, 32000, 8000, True),
              Model('qwen/qwen3-coder', regular, 32000, 8000, True),
              Model('anthropic/claude-sonnet-4.6', regular, 32000, 8000, True),
              Model('fixture/no-tools', cheap, 32000, 8000, False),
              Model('fixture/unknown-price', None, 32000, 8000, True)]
    result = SimpleNamespace(provider='openrouter', models=models, updated_at='fixture', status='fixture', warning=None, source='offline fixture')
    result.find = lambda identifier: next((m for m in models if m.id == identifier), None)
    return result


class RoutingTests(unittest.TestCase):
    def test_three_levels_english_and_russian_with_reasons(self):
        self.assertEqual(complexity('Опиши файл')['level'], 'simple')
        self.assertEqual(complexity('Сделай рефакторинг и тесты')['level'], 'medium')
        self.assertEqual(complexity('Fix security race condition')['level'], 'complex')
        self.assertTrue(complexity('Перенеси архитектуру')['reasons'])

    def test_tasks_and_long_requests_increase_score(self):
        self.assertEqual(complexity('Inspect', tasks=True)['level'], 'medium')
        self.assertGreater(complexity('x' * 2000)['score'], complexity('x')['score'])

    def test_model_routing_uses_capability_pools(self):
        catalog = fixture_catalog()
        for request, expected in (('Опиши файл', 'deepseek/deepseek-v4.1-flash'),
                                  ('Сделай рефакторинг и тесты', 'qwen/qwen3-coder'),
                                  ('Исправь уязвимость безопасности', 'anthropic/claude-sonnet-4.6')):
            self.assertEqual(choose_route(catalog, request)[0].id, expected)

    def test_context_output_and_tools_are_checked_before_selection(self):
        catalog = fixture_catalog()
        for identifier in ('fixture/no-tools', 'fixture/unknown-price'):
            with self.assertRaises(PricingError): choose_route(catalog, 'Inspect', overrides={'simple': identifier})
        with self.assertRaises(PricingError): choose_route(catalog, 'Inspect', max_tokens=9000)
        catalog.models[0] = Model(catalog.models[0].id, catalog.models[0].price, 100, 8000, True)
        with self.assertRaises(PricingError): choose_route(catalog, 'Inspect', overrides={'simple': catalog.models[0].id})

    def test_missing_pool_does_not_silently_send_to_another_model(self):
        catalog = fixture_catalog(); catalog.models.clear()
        with self.assertRaises(PricingError): choose_route(catalog, 'Fix security')

    def test_operator_pool_override(self):
        model, route = choose_route(fixture_catalog(), 'Fix security', overrides={'complex': 'qwen/qwen3-coder'})
        self.assertEqual(model.id, 'qwen/qwen3-coder')
        self.assertEqual(route['level'], 'complex')

    def test_picker_lists_prices_then_validates_input_without_default_consent(self):
        output = []
        answers = iter(['bad', '99', '2'])
        model = pick_model(fixture_catalog(), read=lambda prompt: next(answers), show=output.append)
        self.assertEqual(model.id, 'qwen/qwen3-coder')
        self.assertTrue(any('$/1M' in text for text in output))
        self.assertFalse(any('fixture/no-tools' in text for text in output))

    def test_picker_noninteractive_empty_zero_and_eof_do_not_choose(self):
        with self.assertRaises(PricingError): pick_model(fixture_catalog(), interactive=False)
        for value in ('', '0'):
            with self.assertRaises(PricingError): pick_model(fixture_catalog(), read=lambda prompt: value, show=lambda text: None)
        with self.assertRaises(PricingError): pick_model(fixture_catalog(), read=lambda prompt: (_ for _ in ()).throw(EOFError()), show=lambda text: None)

    def test_ollama_routes_only_installed_tool_models(self):
        catalog = fixture_catalog(); catalog.provider = 'ollama'
        zero = Price('0', '0')
        catalog.models[:] = [Model('qwen:4b', zero, 32000, None, True), Model('qwen:14b', zero, 32000, None, True)]
        self.assertEqual(choose_route(catalog, 'Inspect')[0].id, 'qwen:4b')
        self.assertEqual(choose_route(catalog, 'Fix security')[0].id, 'qwen:14b')

    def test_explicit_cli_model_overrides_env_routing_without_catalog(self):
        args = SimpleNamespace(action='run', provider='openrouter', model='fixed', route=True)
        settings = Settings('openrouter', 'fixed', 'https://openrouter.ai/api/v1', 'key')
        with patch('hw_ai_agent.cli.Settings.from_env', return_value=settings), patch('hw_ai_agent.cli.catalog_for') as catalog:
            self.assertEqual(choose_settings(args).model, 'fixed'); catalog.assert_not_called()

    def test_auto_selector_and_route_flag_select_before_model_generation(self):
        args = SimpleNamespace(action='run', provider='openrouter', model='auto', route=False, request=['Fix security'], max_tokens=2000)
        settings = Settings('openrouter', 'auto', 'https://openrouter.ai/api/v1', 'key')
        with patch('hw_ai_agent.cli.Settings.from_env', return_value=settings), patch('hw_ai_agent.cli.catalog_for', return_value=fixture_catalog()), redirect_stdout(io.StringIO()):
            selected = choose_settings(args)
        self.assertEqual(selected.model, 'anthropic/claude-sonnet-4.6')
        self.assertEqual(args._routing['level'], 'complex')

    def test_routing_metadata_survives_resume_without_reclassification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); workspace = root / 'workspace'; workspace.mkdir()
            (workspace / 'value.py').write_text('value=1')
            settings = Settings('openrouter', 'qwen/qwen3-coder', 'https://openrouter.ai/api/v1', 'key')
            _, route = choose_route(fixture_catalog(), 'Refactor and test')
            class Client:
                max_tokens = 2048
                def __init__(self, first):
                    self.settings, self.routing, self.first = settings, route, first
                def complete(self, messages, tools):
                    return {'tool_calls': [{'id': 'read', 'function': {'name': 'read_file', 'arguments': json.dumps({'path': 'value.py'})}}]} if self.first else {'content': 'done'}
            def pause(event):
                if event['event'] == 'tool_result': raise KeyboardInterrupt
            with SessionStore(root / 'state') as store:
                agent = CodingAgent(Client(True), ToolRuntime(workspace), store=store, on_event=pause)
                try: agent.run('Refactor and test')
                except KeyboardInterrupt: pass
                self.assertEqual(store.get(agent.session_id).metadata['routing']['selected_model'], settings.model)
                with patch('hw_ai_agent.cli.choose_route', side_effect=AssertionError('Do not reroute')):
                    CodingAgent(Client(False), ToolRuntime(workspace), store=store).resume(agent.session_id)
                self.assertEqual(store.get(agent.session_id).metadata['model'], settings.model)

    def test_models_command_is_json_and_does_not_need_api_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = ['hw-ai-agent', 'models', '--provider', 'openrouter', '--state-db', str(Path(temporary)/'state'), '--env-file', str(Path(temporary)/'missing'), '--json']
            with patch('sys.argv', args), patch.dict(os.environ, {}, clear=True), patch('hw_ai_agent.cli.catalog_for', return_value=fixture_catalog()), patch('hw_ai_agent.cli.OpenAICompatibleClient') as client, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(), 0)
            data = json.loads(output.getvalue())
            self.assertEqual(data['currency'], 'USD')
            self.assertIsNone(next(m for m in data['models'] if m['id']=='fixture/unknown-price')['scenario_usd'])
            client.assert_not_called()

    def test_model_question_requires_terminal_before_catalog_or_api(self):
        args = SimpleNamespace(action='run', provider='openrouter', model='?', route=False)
        with patch('hw_ai_agent.cli.Settings.from_env', return_value=Settings('openrouter', '?', 'https://openrouter.ai/api/v1', 'key')), patch('sys.stdin.isatty', return_value=False), patch('hw_ai_agent.cli.catalog_for') as catalog:
            with self.assertRaises(PricingError): choose_settings(args)
            catalog.assert_not_called()

    def test_cli_auto_run_saves_selected_model_cost_and_does_not_reroute_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); workspace = root / 'workspace'; workspace.mkdir()
            db = root / 'state'
            class Client:
                def __init__(self, settings, **kwargs):
                    self.settings, self.max_tokens, self.usage = settings, kwargs['max_tokens'], []
                def complete(self, messages, tools):
                    self.usage.append({'cost': '0.003'})
                    return {'content': 'done'}
            def settings(provider, model):
                return Settings('openrouter', model or 'base', 'https://openrouter.ai/api/v1', 'key')
            with patch.dict(os.environ, {}, clear=True), patch('hw_ai_agent.cli.Settings.from_env', side_effect=settings), patch('hw_ai_agent.cli.catalog_for', return_value=fixture_catalog()), patch('hw_ai_agent.cli.OpenAICompatibleClient', side_effect=Client), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with patch('sys.argv', ['hw-ai-agent', 'run', '--model', 'auto', '--env-file', str(root/'missing'), '--workspace', str(workspace), '--state-db', str(db), 'Fix security']):
                    self.assertEqual(main(), 0)
                with SessionStore(db) as store:
                    identifier = store.list_sessions()[0]['id']
                    self.assertEqual(store.get(identifier).metadata['model'], 'anthropic/claude-sonnet-4.6')
                    before = store.cost(identifier)
                    self.assertEqual(before['known_usd'], '0.003')
                with patch('sys.argv', ['hw-ai-agent', 'resume', identifier, '--env-file', str(root/'missing'), '--state-db', str(db)]), patch('hw_ai_agent.cli.choose_route', side_effect=AssertionError('Never reroute resume')):
                    self.assertEqual(main(), 0)
                with SessionStore(db) as store: self.assertEqual(store.cost(identifier), before)

    def test_cli_question_selects_concrete_id_before_client_creation(self):
        args = SimpleNamespace(action='run', provider='openrouter', model='?', route=False, max_tokens=2000)
        settings = Settings('openrouter', '?', 'https://openrouter.ai/api/v1', 'key')
        with patch('hw_ai_agent.cli.Settings.from_env', return_value=settings), patch('sys.stdin.isatty', return_value=True), patch('hw_ai_agent.cli.catalog_for', return_value=fixture_catalog()), patch('builtins.input', return_value='2'), redirect_stdout(io.StringIO()):
            self.assertEqual(choose_settings(args).model, 'qwen/qwen3-coder')


if __name__ == '__main__': unittest.main()
