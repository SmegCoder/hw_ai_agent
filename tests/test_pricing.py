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

from hw_ai_agent.agent import CodingAgent, AgentError
from hw_ai_agent.client import OpenAICompatibleClient, ModelClientError
from hw_ai_agent.config import Settings
from hw_ai_agent.cost_demo import run_cost_demo
from hw_ai_agent.model_catalog import ModelCatalog, openai_models, openrouter_models, comparison
from hw_ai_agent.pricing import Price, PricingError, amount, usd, estimate, estimate_tokens, actual_cost, cost_summary
from hw_ai_agent.session import SessionStore
from hw_ai_agent.tools import ToolRuntime
from hw_ai_agent.cli import main, configure_client


ROUTER = {'data': [{'id': 'fixture/small', 'pricing': {'prompt': '0.000001', 'completion': '0.000002'}, 'context_length': 32000, 'supported_parameters': ['tools'], 'architecture': {'output_modalities': ['text']}}]}
OPENAI = '''### Standard pricing data
| Model | In | Cache | Write | Out | Long in | Long cache | Long write | Long out |
| gpt-4.1-mini | $0.40 | $0.10 | - | $1.60 | - | - | - | - |
| gpt-5.4 | $2.50 | $0.25 | - | $15.00 | $5.00 | $0.50 | - | $22.50 |
### Batch pricing data
| gpt-4.1-mini | $0.20 | - | - | $0.80 | - | - | - | - |
'''


class PriceTests(unittest.TestCase):
    def setUp(self):
        self.price = Price('0.000001', '0.000002', cached='0.0000005', cache_write='0.00000125')

    def test_decimal_cost_including_cache_and_writes(self):
        self.assertEqual(self.price.calculate(1000, 100, 200, 100), Decimal('0.001125'))

    def test_unknown_is_distinct_from_zero_and_invalid_numbers(self):
        for value in (None, True, False, -1, 'NaN', 'Infinity', {}, 'bad'):
            self.assertIsNone(amount(value))
        self.assertEqual(usd(None), 'неизвестно')
        self.assertEqual(usd('0'), '$0.00000000')
        with self.assertRaises(PricingError): Price(None, '0')

    def test_cache_without_rate_and_bad_counts_are_unknown(self):
        price = Price('0.000001', '0.000002')
        self.assertIsNone(price.calculate(100, 10, cached=10))
        self.assertIsNone(price.calculate(100, 10, cached=101))
        self.assertIsNone(price.calculate(True, 10))

    def test_reported_cost_is_authoritative_even_when_truncated(self):
        record = actual_cost('openrouter', self.price, {'cost': '0.0004', 'prompt_tokens': 1000, 'completion_tokens': 100}, failed=True)
        self.assertEqual((record['kind'], record['usd']), ('reported', '0.0004'))

    def test_direct_openai_calculation_uses_actual_tokens(self):
        record = actual_cost('openai', self.price, {'prompt_tokens': 1000, 'completion_tokens': 100, 'prompt_tokens_details': {'cached_tokens': 200, 'cache_write_tokens': 100}})
        self.assertEqual(record['kind'], 'calculated')
        self.assertEqual(Decimal(record['usd']), Decimal('0.001125'))

    def test_missing_usage_network_failures_and_other_tier_are_unknown(self):
        self.assertEqual(actual_cost('openrouter', self.price, {}, failed=True)['kind'], 'unknown')
        self.assertEqual(actual_cost('openai', self.price, {'prompt_tokens': 1, 'completion_tokens': 1, '_nonstandard_tier': True})['kind'], 'unknown')

    def test_local_api_cost_is_zero_not_electricity_cost(self):
        self.assertEqual(actual_cost('ollama', None, {})['kind'], 'local')
        self.assertEqual(actual_cost('scripted', None, {})['usd'], '0')

    def test_quote_includes_tools_and_complete_history(self):
        messages = [{'role': 'user', 'content': 'Привет'}]
        base = estimate_tokens(messages, [])
        tools = [{'function': {'name': 'read', 'parameters': {'description': 'x' * 500}}}]
        self.assertGreater(estimate_tokens(messages, tools), base)
        self.assertGreater(estimate_tokens(messages + [{'role': 'tool', 'content': 'x' * 1000}], tools), base)
        quote = estimate(self.price, messages, tools, 200)
        self.assertLess(Decimal(quote['usd_input_estimate']), Decimal(quote['usd_with_full_output_estimate']))

    def test_long_context_prices_not_short_context(self):
        price = openai_models(OPENAI, 'now')[1].price
        self.assertEqual(price.calculate(300000, 100), Decimal('1.5022500'))

    def test_summary_separates_billed_calculated_and_unknown(self):
        events = [{'event': 'model_cost', 'round': 1, 'usd': '0.1', 'kind': 'reported'},
                  {'event': 'model_cost', 'round': 2, 'usd': '0.2', 'kind': 'calculated'}]
        result = cost_summary(events, 3)
        self.assertEqual(result['known_usd'], '0.3')
        self.assertEqual(result['reported_usd'], '0.1')
        self.assertEqual(result['unknown_requests'], 1)
        self.assertFalse(result['complete'])

    def test_conflicting_duplicate_costs_do_not_double_charge(self):
        event = {'event': 'model_cost', 'round': 1, 'usd': '0.1', 'kind': 'reported'}
        self.assertEqual(cost_summary([event, event])['known_usd'], '0.1')
        with self.assertRaises(PricingError): cost_summary([event, {**event, 'usd': '0.2'}])


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_router_units_and_unknown_price_are_preserved(self):
        models = openrouter_models({'data': ROUTER['data'] + [{'id': 'fixture/unknown', 'supported_parameters': ['tools']}, 'junk']}, 'now')
        self.assertEqual(models[0].price.calculate(1000, 1000), Decimal('0.003000'))
        self.assertIsNone(models[1].price)
        self.assertIn('неизвестно', comparison(models))

    def test_openai_uses_standard_and_converts_per_million(self):
        model = openai_models(OPENAI, 'now')[0]
        self.assertEqual(model.price.input, '4E-7')
        self.assertEqual(model.price.calculate(1000, 1000), Decimal('0.0020000'))
        with self.assertRaises(PricingError): openai_models('### Batch pricing data', 'now')

    def test_cache_fresh_stale_and_refresh(self):
        with patch.object(ModelCatalog, '_get', return_value=json.dumps(ROUTER)) as network:
            first = ModelCatalog('openrouter', self.root).load()
            second = ModelCatalog('openrouter', self.root).load()
            self.assertEqual((first.status, second.status), ('live', 'cache'))
            self.assertEqual(network.call_count, 1)
            ModelCatalog('openrouter', self.root).load(refresh=True)
            self.assertEqual(network.call_count, 2)
        with patch.object(ModelCatalog, '_get', side_effect=PricingError('down')):
            stale = ModelCatalog('openrouter', self.root, ttl=-1).load()
            self.assertEqual(stale.status, 'stale'); self.assertIsNotNone(stale.warning)

    def test_no_cache_failure_does_not_invent_zero_prices(self):
        with patch.object(ModelCatalog, '_get', side_effect=PricingError('down')):
            with self.assertRaises(PricingError): ModelCatalog('openrouter', self.root).load()

    def test_corrupt_cache_is_refetched(self):
        (self.root / 'models-openrouter.json').write_text('{bad')
        with patch.object(ModelCatalog, '_get', return_value=json.dumps(ROUTER)):
            self.assertEqual(ModelCatalog('openrouter', self.root).load().status, 'live')

    def test_public_catalog_never_sends_api_key(self):
        catalog = ModelCatalog('openrouter', self.root)
        with patch.object(catalog.opener, 'open', return_value=io.BytesIO(json.dumps(ROUTER).encode())) as transport:
            catalog.load()
        self.assertNotIn('Authorization', transport.call_args.args[0].headers)

    def test_batch_only_variants_are_not_chat_tool_candidates(self):
        row = {**ROUTER['data'][0], 'id': 'fixture/small:batch'}
        self.assertFalse(openrouter_models({'data': [row]}, 'now')[0].tools)

    def test_ollama_lists_only_installed_models_and_checks_tools(self):
        def get(url, data=None):
            return json.dumps({'models': [{'name': 'qwen:4b'}]} if url.endswith('/tags') else {'capabilities': ['tools'], 'model_info': {'qwen.context_length': 32000}})
        with patch.object(ModelCatalog, '_get', side_effect=get):
            catalog = ModelCatalog('ollama', self.root, base_url='http://localhost:11434/v1').load()
        self.assertTrue(catalog.models[0].tools)
        self.assertEqual(catalog.models[0].price.input, '0')


class AccountingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.workspace = self.root / 'workspace'; self.workspace.mkdir()

    def test_usage_saved_before_truncation_error(self):
        client = OpenAICompatibleClient(Settings('openrouter', 'fixture', 'https://example.org/v1', 'private-key'))
        body = {'choices': [{'finish_reason': 'length', 'message': {'content': 'cut'}}], 'usage': {'cost': 0.005, 'prompt_tokens': 10, 'completion_tokens': 5}}
        with SessionStore(self.root / 'state') as store, patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps(body).encode())):
            agent = CodingAgent(client, ToolRuntime(self.workspace), store=store)
            with self.assertRaises(ModelClientError): agent.run('Task')
            self.assertEqual(store.cost(agent.session_id)['reported_usd'], '0.005')
            self.assertEqual(store.cost(agent.session_id)['unknown_requests'], 0)

    def test_local_scripted_client_quote_and_total_are_zero(self):
        client = SimpleNamespace(billing_provider='scripted', complete=lambda messages, tools: {'content': 'ok'})
        agent = CodingAgent(client, ToolRuntime(self.workspace))
        agent.run('Task')
        quote = next(e for e in agent.events if e['event']=='model_cost_estimate')
        self.assertEqual(quote['usd_with_full_output_estimate'], '0')
        self.assertEqual(cost_summary(agent.events)['known_usd'], '0')
        self.assertTrue(cost_summary(agent.events)['complete'])

    def test_interrupt_after_paid_response_preserves_charge(self):
        client = OpenAICompatibleClient(Settings('openrouter', 'fixture', 'https://example.org/v1', 'key'))
        body = {'choices': [{'message': {'content': 'ok'}}], 'usage': {'cost': '0.004'}}
        complete = client.complete
        def interrupt(messages, tools):
            complete(messages, tools)
            raise KeyboardInterrupt
        with SessionStore(self.root / 'state') as store, patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps(body).encode())), patch.object(client, 'complete', side_effect=interrupt):
            agent = CodingAgent(client, ToolRuntime(self.workspace), store=store)
            with self.assertRaises(KeyboardInterrupt): agent.run('Task')
            self.assertEqual(store.cost(agent.session_id)['reported_usd'], '0.004')
            self.assertEqual(store.get(agent.session_id).status, 'interrupted')

    def test_transport_failure_never_reuses_previous_usage(self):
        client = OpenAICompatibleClient(Settings('openrouter', 'fixture', 'https://example.org/v1', 'private-key'))
        client.usage.append({'cost': 0.5})
        with SessionStore(self.root / 'state') as store, patch.object(client.opener, 'open', side_effect=OSError('down')):
            agent = CodingAgent(client, ToolRuntime(self.workspace), store=store)
            with self.assertRaises(ModelClientError): agent.run('Task')
            result = store.cost(agent.session_id)
            self.assertEqual(result['known_usd'], '0'); self.assertEqual(result['unknown_requests'], 1)

    def test_provider_charge_saved_even_if_protocol_rejects_answer(self):
        client = OpenAICompatibleClient(Settings('openrouter', 'fixture', 'https://example.org/v1', 'private-key'))
        body = {'choices': [{'message': {'content': ''}}], 'usage': {'cost': 0.003}}
        with SessionStore(self.root / 'state') as store, patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps(body).encode())):
            agent = CodingAgent(client, ToolRuntime(self.workspace), store=store)
            with self.assertRaises(AgentError): agent.run('Task')
            self.assertEqual(store.cost(agent.session_id)['reported_usd'], '0.003')

    def test_unrecorded_round_after_crash_is_unknown_and_legacy_supported(self):
        with SessionStore(self.root / 'state') as store:
            agent = CodingAgent(SimpleNamespace(complete=lambda messages, tools: {'content': 'ok'}), ToolRuntime(self.workspace), store=store)
            agent.run('Task'); store.start_model(agent.session_id)
            self.assertEqual(store.cost(agent.session_id)['unknown_requests'], 2)
            self.assertEqual(store.db.execute('PRAGMA user_version').fetchone()[0], 5)

    def test_cached_completed_resume_does_not_add_charge(self):
        client = OpenAICompatibleClient(Settings('openrouter', 'fixture', 'https://example.org/v1', 'private-key'))
        body = {'choices': [{'message': {'content': 'ok'}}], 'usage': {'cost': 0.003}}
        with SessionStore(self.root / 'state') as store, patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps(body).encode())) as network:
            agent = CodingAgent(client, ToolRuntime(self.workspace), store=store)
            agent.run('Task'); before = store.cost(agent.session_id)
            agent.resume(agent.session_id)
            self.assertEqual(store.cost(agent.session_id), before); self.assertEqual(network.call_count, 1)

    def test_cost_cli_requires_no_key_and_does_not_start_model(self):
        with SessionStore(self.root / 'state') as store:
            agent = CodingAgent(SimpleNamespace(complete=lambda messages, tools: {'content': 'ok'}), ToolRuntime(self.workspace), store=store); agent.run('Task')
            identifier = agent.session_id
        with patch('sys.argv', ['hw-ai-agent', 'cost', identifier, '--state-db', str(self.root / 'state')]), patch('hw_ai_agent.cli.OpenAICompatibleClient') as client, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(), 0)
            self.assertEqual(json.loads(out.getvalue())['cost']['unknown_requests'], 1)
            client.assert_not_called()

    def test_offline_cost_demo_checks_resume_and_routing(self):
        report = run_cost_demo()
        self.assertTrue(report['success']); self.assertEqual(report['cost']['known_usd'], '0.0030005')

    def test_custom_endpoint_does_not_borrow_official_rates(self):
        settings = Settings('openai', 'gpt-4.1', 'https://custom.example/v1', 'key')
        client = OpenAICompatibleClient(settings)
        with patch('hw_ai_agent.cli.catalog_for') as catalog, redirect_stdout(io.StringIO()):
            configure_client(client, settings, SimpleNamespace())
        self.assertIsNone(client.price); catalog.assert_not_called()


if __name__ == '__main__': unittest.main()
