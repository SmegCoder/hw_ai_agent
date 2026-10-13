from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .agent import CodingAgent
from .model_catalog import Model
from .pricing import Price, actual_cost, estimate
from .routing import choose_route
from .session import SessionStore
from .tools import ToolRuntime


def run_cost_demo(on_event=None):
    price = Price('0.000001', '0.000002', cached='0.0000005', source='offline fixture', updated_at='fixture')
    checks = {}
    quote = estimate(price, [{'role': 'user', 'content': 'Прочитай файл'}], [], 100)
    checks['Estimate before answer'] = quote['usd_with_full_output_estimate'] is not None
    checks['Reported cost overrides estimate'] = actual_cost('openrouter', price, {'cost': 0.004})['usd'] == '0.004'
    checks['Unknown is not zero'] = actual_cost('openrouter', None, {})['usd'] is None
    class Client:
        settings = SimpleNamespace(provider='openrouter', model='fixture-model', base_url='offline://cost-demo')
        max_tokens = 100
        def __init__(self, first):
            self.first, self.usage, self.price = first, [], price
        def complete(self, messages, tools):
            self.usage.append({'prompt_tokens': 100, 'completion_tokens': 20, 'cost': '0.0010005' if self.first else '0.002'})
            if self.first:
                return {'tool_calls': [{'id': 'read', 'function': {'name': 'read_file', 'arguments': json.dumps({'path': 'value.py'})}}]}
            return {'content': 'Файл изучен.'}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); workspace = root / 'workspace'; workspace.mkdir()
        (workspace / 'value.py').write_text('value = 1\n')
        db = root / 'state.sqlite3'
        def pause(event):
            if on_event: on_event(event)
            if event['event'] == 'tool_result': raise KeyboardInterrupt
        with SessionStore(db) as store:
            agent = CodingAgent(Client(True), ToolRuntime(workspace), store=store, on_event=pause)
            try: agent.run('Прочитай файл')
            except KeyboardInterrupt: pass
            identifier = agent.session_id
            before = store.cost(identifier)
        with SessionStore(db) as store:
            CodingAgent(Client(False), ToolRuntime(workspace), store=store, on_event=on_event).resume(identifier)
            after = store.cost(identifier)
            checks['Resume retains exact decimal sum'] = before['known_usd'] == '0.0010005' and after['known_usd'] == '0.0030005'
            checks['No unknown requests after clean resume'] = after['complete'] and after['requests'] == 2
    catalog = SimpleNamespace(provider='openrouter', updated_at='offline fixture', status='fixture')
    catalog.models = [Model('deepseek/deepseek-v4.1-flash', price, 100000, 10000, True),
                      Model('qwen/qwen3-coder', price, 100000, 10000, True),
                      Model('anthropic/claude-sonnet-4.6', price, 100000, 10000, True)]
    catalog.find = lambda identifier: next((m for m in catalog.models if m.id == identifier), None)
    routes = []
    for request in ('Опиши файл', 'Сделай рефакторинг и добавь тесты', 'Исправь уязвимость безопасности и гонку'):
        model, result = choose_route(catalog, request)
        routes.append({'request': request, 'model': model.id, 'level': result['level']})
    checks['Three task complexity levels'] = [r['level'] for r in routes] == ['simple', 'medium', 'complex']
    return {'success': all(checks.values()), 'checks': checks, 'routes': routes, 'cost': after,
            'note': 'Offline fixture prices and responses; no paid API requests.'}
