from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from .agent import CodingAgent, AgentError
from .client import ModelClientError
from .demo import initialize_demo, verify_demo
from .budget import BudgetExceeded
from .tools import ToolRuntime
from .session import SessionStore
from .workflow_demo import call, CrashAfterWriteRuntime


FIXED_ORDER = "def calculate_total(prices, discount_percent):\n    if not 0 <= discount_percent <= 100:\n        raise ValueError('invalid discount')\n    return sum(prices) * (1 - discount_percent / 100)\n"
DEMO_PLAN = {'goal': 'Исправить расчёт стоимости заказа без изменения тестов',
             'steps': [{'description': 'Исправить формулу и проверку диапазона скидки', 'tools': ['write_file']},
                       {'description': 'Запустить контрольные тесты', 'tools': ['Bash']}],
             'risks': ['Не изменять контрольный тестовый файл'],
             'assumptions': ['Допустимая скидка — от 0 до 100 процентов'],
             'done_criteria': ['Шесть тестов проходят', 'Тестовый файл не изменён']}


class PlanDemoClient:
    settings = SimpleNamespace(provider='scripted', model='plan-demo', base_url='offline://plan-demo')
    usage = []
    max_tokens = 2048

    def complete(self, messages, tools):
        results = [m for m in messages if m.get('role') == 'tool']
        if not results:
            return {'tool_calls': [call('list', 'list_files', {'path': '.'}),
                                   call('read', 'read_file', {'path': 'order.py'}),
                                   call('tests-read', 'read_file', {'path': 'test_order.py'})]}
        if not any(m.get('name') == 'submit_plan' and not m['content'].startswith('ERROR:') for m in results):
            return {'tool_calls': [call('plan', 'submit_plan', DEMO_PLAN)]}
        if not any(m.get('tool_call_id') == 'tests' for m in results):
            return {'tool_calls': [call('write', 'write_file', {'path': 'order.py', 'content': FIXED_ORDER}),
                                   call('tests', 'Bash', {'command': 'python -m unittest discover -v'})]}
        return {'content': 'План исполнен. Результат проверяется независимым запуском тестов.'}


def run_plan_demo(workspace: Path, db_path: Path, *, mode='safe', approve=None,
                  approve_plan=None, on_event=None, crash_after_write=False,
                  client=None, offline=False, max_rounds=20, approvals='ask', policy=None,
                  session_metadata=None, budget_limits=None, journal_mode=False):
    if max_rounds < 1:
        raise ValueError('Limits must be positive')
    if offline:
        if client is not None:
            raise ValueError('Offline demo cannot use a real model client')
        client = PlanDemoClient()
    elif client is None:
        raise ValueError('Real model client required; use offline=True for scripted demo')
    if db_path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError('Session database must be outside workspace')
    initialize_demo(workspace)
    baseline = 'Baseline not run: Bash is disabled until plan approval.'
    test_hash = hashlib.sha256((workspace / 'test_order.py').read_bytes()).hexdigest()
    cls = CrashAfterWriteRuntime if crash_after_write else ToolRuntime
    runtime = cls(workspace, mode=mode, approve=approve, policy=policy, trash_dir=db_path.resolve().parent / 'trash')
    with SessionStore(db_path) as store:
        agent = CodingAgent(client, runtime, plan_mode=True, approve_plan=approve_plan, store=store,
                            max_rounds=max_rounds, on_event=on_event, budget_limits=budget_limits, journal_mode=journal_mode,
                            session_metadata={**(session_metadata or {}), 'kind': 'plan-demo', 'approvals': approvals,
                                              'demo': {'baseline': baseline, 'test_hash': test_hash, 'offline': offline}})
        error = None
        try:
            answer = agent.run('Изучи проект и составь план исправления расчёта итоговой стоимости заказа. '
                               'После утверждения плана исправь код и запусти тесты через Bash. '
                               'Тесты не меняй; не создавай вспомогательные файлы. Отвечай по-русски.')
        except (AgentError, BudgetExceeded, ModelClientError) as exc:
            if runtime.phase != 'execution':
                raise
            error, answer = str(exc), ''
        return verify_demo(client, workspace, runtime, baseline, test_hash, answer, agent.events,
                           error=error, session_id=agent.session_id)
