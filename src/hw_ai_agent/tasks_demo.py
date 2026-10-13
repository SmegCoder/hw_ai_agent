from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .agent import CodingAgent, AgentError
from .budget import BudgetExceeded
from .client import ModelClientError
from .session import SessionStore
from .tools import ToolRuntime
from .tasks import checklist
from .workflow_demo import CrashAfterWriteRuntime
from .reporting import public_report


TAX_BROKEN = 'def add_tax(amount):\n    return amount + 20\n'
RECEIPT_BROKEN = 'from tax import add_tax\n\ndef receipt_total(prices):\n    return add_tax(len(prices))\n'
TAX_FIXED = 'def add_tax(amount):\n    return amount * 1.2\n'
RECEIPT_FIXED = 'from tax import add_tax\n\ndef receipt_total(prices):\n    return add_tax(sum(prices))\n'
TESTS = '''import unittest
from tax import add_tax
from receipt import receipt_total
class CheckoutTests(unittest.TestCase):
    def test_tax(self):
        self.assertAlmostEqual(add_tax(50), 60)
    def test_zero_tax(self):
        self.assertEqual(add_tax(0), 0)
    def test_receipt(self):
        self.assertAlmostEqual(receipt_total([100,50]), 180)
    def test_empty(self):
        self.assertEqual(receipt_total([]), 0)
'''
TASK_PLAN = {'goal': 'Исправить расчёт чека с налогом 20%', 'steps': [
    {'id': 'tax', 'description': 'Исправить процентный налог', 'tools': ['read_file', 'write_file'], 'depends_on': [],
     'verification': {'kind': 'file_written', 'path': 'tax.py'}},
    {'id': 'receipt', 'description': 'Исправить суммирование цен в чеке', 'tools': ['read_file', 'write_file'], 'depends_on': ['tax'],
     'verification': {'kind': 'file_written', 'path': 'receipt.py'}},
    {'id': 'tests', 'description': 'Проверить оба исправления тестами', 'tools': ['Bash'], 'depends_on': ['tax', 'receipt'],
     'verification': {'kind': 'tests_passed', 'command': 'python -m unittest discover -v', 'cwd': '.'}}
], 'risks': ['Тесты нельзя изменять'], 'assumptions': ['Налог составляет 20%'],
'done_criteria': ['Четыре контрольных теста проходят', 'Исходные тесты не изменены']}


def call(identifier, name, **args):
    return {'id': identifier, 'function': {'name': name, 'arguments': json.dumps(args)}}


class TasksDemoClient:
    settings = SimpleNamespace(provider='scripted', model='tasks-demo', base_url='offline://tasks-demo')
    usage = []
    max_tokens = 3072

    def complete(self, messages, tools):
        schemas = {s['function']['name'] for s in tools}
        if 'submit_plan' in schemas:
            results = [m for m in messages if m.get('role') == 'tool']
            if not results:
                return {'tool_calls': [call('p_list', 'list_files', path='.'), call('p_tax', 'read_file', path='tax.py'),
                                       call('p_receipt', 'read_file', path='receipt.py'), call('p_tests', 'read_file', path='test_checkout.py')]}
            return {'tool_calls': [call('plan', 'submit_plan', **TASK_PLAN)]}
        if not tools:
            return {'content': 'Все три задачи завершены; результаты подтверждены runtime.'}
        current = next(m for m in reversed(messages) if m.get('role') == 'user' and m.get('content', '').startswith('CURRENT_TASK\n'))
        task = json.loads(current['content'].split('\n', 1)[1])['id']
        results = {m.get('tool_call_id'): m for m in messages if m.get('role') == 'tool'}
        if task in {'tax', 'receipt'}:
            action = 'write_' + task
            if action in results and results[action]['content'].startswith('ERROR:'):
                return {'tool_calls': [call('fail_' + task, 'fail_task', task_id=task, reason='Запись не разрешена или не подтверждена после сбоя')]}
            if action not in results:
                return {'tool_calls': [call('read_' + task, 'read_file', path=task + '.py'),
                                       call(action, 'write_file', path=task + '.py', content=TAX_FIXED if task == 'tax' else RECEIPT_FIXED),
                                       call('finish_' + task, 'complete_task', task_id=task)]}
            return {'tool_calls': [call('finish_' + task, 'complete_task', task_id=task)]}
        return {'tool_calls': [call('run_tests', 'Bash', command='python -m unittest discover -v'),
                               call('finish_tests', 'complete_task', task_id='tests')]}


def run_tasks_demo(workspace: Path, db_path: Path, *, client=None, offline=False, mode='safe', approve=None,
                   approve_plan=None, on_event=None, max_rounds=30, budget_limits=None, policy=None,
                   session_metadata=None, approvals='ask', crash_after_write=False, max_tokens=3072):
    if offline:
        if client is not None:
            raise ValueError('Offline demo cannot use a real model client')
        client = TasksDemoClient()
        client.max_tokens = max_tokens
    elif client is None:
        raise ValueError('Real model client required; use offline=True for deterministic demo')
    if db_path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError('Session database must be outside workspace')
    workspace.mkdir(parents=True, exist_ok=False)
    for name, content in [('tax.py', TAX_BROKEN), ('receipt.py', RECEIPT_BROKEN), ('test_checkout.py', TESTS)]:
        (workspace / name).write_text(content, encoding='utf-8')
    runtime_class = CrashAfterWriteRuntime if crash_after_write else ToolRuntime
    runtime = runtime_class(workspace, mode=mode, policy=policy, approve=approve, trash_dir=db_path.resolve().parent / 'trash')
    metadata = {**(session_metadata or {}), 'kind': 'tasks-demo', 'approvals': approvals,
                'demo': {'offline': offline, 'test_hash': hashlib.sha256(TESTS.encode()).hexdigest()}}
    with SessionStore(db_path) as store:
        agent = CodingAgent(client, runtime, task_mode=True, store=store, approve_plan=approve_plan,
                            on_event=on_event, max_rounds=max_rounds, budget_limits=budget_limits, session_metadata=metadata)
        error, answer = None, ''
        try:
            answer = agent.run('Изучи файлы и составь план минимум из трёх задач с зависимостями и проверяемыми результатами. '
                               'Исправь расчёт налога и общей суммы чека, запусти контрольные тесты. Налог — 20% от суммы. '
                               'Тесты не меняй, вспомогательные файлы не создавай. Закрой каждую задачу через complete_task. '
                               'Ответь по-русски.')
        except (AgentError, BudgetExceeded, ModelClientError) as exc:
            if runtime.phase != 'execution':
                raise
            error = str(exc)
        return verify_tasks_demo(client, runtime, store, agent.session_id, answer, metadata, error=error)


def verify_tasks_demo(client, runtime, store, session_id, answer, metadata, error=None):
    # Independent verification belongs to the demo harness, not to the agent budget.
    output = runtime.bash('python -m unittest discover -v')
    tasks = store.tasks(session_id)
    unchanged = hashlib.sha256((runtime.root / 'test_checkout.py').read_bytes()).hexdigest() == metadata['demo']['test_hash']
    complete = len(tasks) >= 3 and all(t['state'] == 'completed' and t['proof'] for t in tasks)
    dependencies = any(t['depends_on'] for t in tasks)
    calls = store.calls(session_id)
    scoped = all((i['phase'] == 'planning' and i['task_id'] is None) or
                 (i['phase'] == 'execution' and i['task_id'] in {t['id'] for t in tasks}) for i in calls)
    verified_tests = any(t['state'] == 'completed' and t['verification']['kind'] == 'tests_passed' for t in tasks)
    report = {'workspace': str(runtime.root), 'session_id': session_id, 'provider': client.settings.provider,
              'model': client.settings.model, 'tasks': tasks, 'checklist': checklist(tasks), 'answer': answer,
              'verification': output, 'tests_unchanged': unchanged, 'task_calls_scoped': scoped,
              'planning_calls': [i for i in calls if i['phase'] == 'planning'],
              'execution_calls': [i for i in calls if i['phase'] == 'execution'],
              'verified_test_task': verified_tests, 'budget': store.budget(session_id), 'error': error,
              'success': error is None and complete and dependencies and scoped and verified_tests and unchanged and output.startswith('exit_code=0\n')}
    (runtime.root / 'tasks-report.json').write_text(json.dumps(public_report(report, runtime.root), ensure_ascii=False, indent=2), encoding='utf-8')
    return report
