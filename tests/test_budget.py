from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.budget import BudgetLimits, BudgetExceeded, DIMENSIONS, truncate_result
from hw_ai_agent.policy import ApprovalDeferred
from hw_ai_agent.session import SessionError, SessionStore, UncertainActionError
from hw_ai_agent.tools import ToolRuntime, ToolError
from hw_ai_agent.plan_demo import DEMO_PLAN


def call(identifier, name, **arguments):
    return {'id': identifier, 'function': {'name': name, 'arguments': json.dumps(arguments)}}


def write(identifier='w', value='fixed'):
    return call(identifier, 'write_file', path=identifier + '.py', content=value)


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(list(messages))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


class Runtime(ToolRuntime):
    def __init__(self, root, **options):
        super().__init__(root, mode=options.pop('mode', 'auto'), **options)
        self.executions = []
        self.quotas = []

    def execute(self, name, arguments):
        self.executions.append(name)
        self.quotas.append(self.result_limit_bytes)
        return super().execute(name, arguments)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        self.path = self.root / '.agent/state.sqlite3'
        self.store = SessionStore(self.path)

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    def agent(self, client, runtime=None, *, models=12, tools=128, effects=32, output=1_000_000, **options):
        return CodingAgent(client, runtime or Runtime(self.workspace), store=self.store,
                           max_rounds=models, budget_limits=BudgetLimits(models, tools, effects, output), **options)

    def spent(self, agent):
        return self.store.budget(agent.session_id)['spent']

    def pause_after_result(self, event):
        if event['event'] == 'tool_result':
            raise KeyboardInterrupt

    def test_model_limit_prevents_next_request_and_persists_after_resume(self):
        client = Client({'tool_calls': [call('ls', 'list_files', path='.') ]})
        agent = self.agent(client, models=1)
        with self.assertRaisesRegex(BudgetExceeded, 'model_requests'):
            agent.run('Task')
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(self.spent(agent)['model_requests'], 1)
        resumed_client = Client()
        with self.assertRaisesRegex(SessionError, 'BUDGET_EXHAUSTED'):
            self.agent(resumed_client, models=99).resume(agent.session_id)
        self.assertEqual(resumed_client.requests, [])
        self.assertEqual(self.store.budget(agent.session_id)['limits']['model_requests'], 1)

    def test_failed_model_request_is_still_charged(self):
        agent = self.agent(Client(RuntimeError('network failed')), models=2)
        with self.assertRaises(RuntimeError):
            agent.run('Task')
        self.assertEqual(self.spent(agent)['model_requests'], 1)
        self.agent(Client({'content': 'done'})).resume(agent.session_id)
        self.assertEqual(self.spent(agent)['model_requests'], 2)

    def test_tool_batch_cannot_execute_even_first_write_when_short(self):
        runtime = Runtime(self.workspace)
        agent = self.agent(Client({'tool_calls': [write(), call('r', 'read_file', path='w.py')]}), runtime, tools=1)
        with self.assertRaisesRegex(BudgetExceeded, 'tool_calls'):
            agent.run('Task')
        self.assertEqual(runtime.executions, [])
        self.assertFalse((self.workspace / 'w.py').exists())
        self.assertEqual(self.spent(agent), dict(model_requests=1, tool_calls=0, side_effects=0, result_bytes=0))
        rejected = self.store.get(agent.session_id).events[-1]
        self.assertEqual(rejected['event'], 'batch_rejected')
        self.assertEqual(len(rejected['assistant']['tool_calls']), 2)

    def test_effect_batch_cannot_execute_first_read_when_short(self):
        runtime = Runtime(self.workspace)
        agent = self.agent(Client({'tool_calls': [call('ls', 'list_files', path='.'), write('a'), write('b')]}), runtime, effects=1)
        with self.assertRaisesRegex(BudgetExceeded, 'side_effects'):
            agent.run('Task')
        self.assertEqual(runtime.executions, [])
        self.assertEqual(self.spent(agent)['tool_calls'], 0)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_output_batch_requires_a_slot_for_every_result(self):
        runtime = Runtime(self.workspace)
        agent = self.agent(Client({'tool_calls': [write(), call('ls', 'list_files', path='.')]}), runtime, output=1)
        with self.assertRaisesRegex(BudgetExceeded, 'result_bytes'):
            agent.run('Task')
        self.assertEqual(runtime.executions, [])
        self.assertEqual(self.spent(agent)['side_effects'], 0)

    def test_zero_tool_limit_refuses_any_tool(self):
        agent = self.agent(Client({'tool_calls': [call('ls', 'list_files', path='.')]}), tools=0)
        with self.assertRaisesRegex(BudgetExceeded, 'tool_calls'):
            agent.run('Task')

    def test_spent_tool_limit_stops_next_batch_without_charging_effect(self):
        runtime = Runtime(self.workspace)
        agent = self.agent(Client({'tool_calls': [call('ls', 'list_files', path='.') ]}, {'tool_calls': [write()]}), runtime, tools=1)
        with self.assertRaisesRegex(BudgetExceeded, 'tool_calls'):
            agent.run('Task')
        self.assertEqual(runtime.executions, ['list_files'])
        self.assertEqual(self.spent(agent)['tool_calls'], 1)
        self.assertEqual(self.spent(agent)['side_effects'], 0)
        self.assertFalse((self.workspace / 'w.py').exists())

    def test_spent_effect_limit_stops_whole_next_batch(self):
        runtime = Runtime(self.workspace)
        agent = self.agent(Client({'tool_calls': [write('a')]}, {'tool_calls': [call('r', 'read_file', path='a.py'), write('b')]}), runtime, effects=1)
        with self.assertRaisesRegex(BudgetExceeded, 'side_effects'):
            agent.run('Task')
        self.assertEqual(runtime.executions, ['write_file'])
        self.assertEqual(self.spent(agent)['tool_calls'], 1)
        self.assertEqual(self.spent(agent)['side_effects'], 1)
        self.assertTrue((self.workspace / 'a.py').exists())
        self.assertFalse((self.workspace / 'b.py').exists())

    def test_zero_effect_limit_still_permits_reading(self):
        (self.workspace / 'a.py').write_text('original')
        agent = self.agent(Client({'tool_calls': [call('r', 'read_file', path='a.py')]}, {'content': 'done'}), effects=0)
        self.assertEqual(agent.run('Task'), 'done')
        self.assertEqual(self.spent(agent)['side_effects'], 0)
        self.assertEqual(self.spent(agent)['result_bytes'], len('original'))

    def test_zero_output_limit_refuses_tools_but_allows_final_text(self):
        agent = self.agent(Client({'tool_calls': [write()]}), output=0)
        with self.assertRaisesRegex(BudgetExceeded, 'result_bytes'):
            agent.run('Task')
        final = self.agent(Client({'content': 'done'}), output=0)
        self.assertEqual(final.run('Task'), 'done')
        self.assertEqual(self.spent(final)['result_bytes'], 0)

    def test_utf8_output_is_bounded_marked_and_charged_by_bytes(self):
        (self.workspace / 'a.py').write_text('Я' * 100)
        runtime = Runtime(self.workspace)
        agent = self.agent(Client({'tool_calls': [call('r', 'read_file', path='a.py')]}, {'content': 'done'}), runtime, output=61)
        agent.run('Task')
        result = self.store.calls(agent.session_id)[0]['result']['content']
        self.assertIn('TRUNCATED', result)
        self.assertLessEqual(len(result.encode('utf-8')), 61)
        self.assertEqual(self.spent(agent)['result_bytes'], len(result.encode('utf-8')))
        self.assertEqual(runtime.quotas, [61])
        self.assertIsNone(runtime.result_limit_bytes)

    def test_output_exhaustion_refuses_next_batch(self):
        (self.workspace / 'a.py').write_text('X' * 100)
        agent = self.agent(Client({'tool_calls': [call('r', 'read_file', path='a.py')]}, {'tool_calls': [write()]}), output=60)
        with self.assertRaisesRegex(BudgetExceeded, 'result_bytes'):
            agent.run('Task')
        self.assertEqual(self.spent(agent)['result_bytes'], 60)
        self.assertFalse((self.workspace / 'w.py').exists())

    def test_large_first_output_leaves_quota_for_rest_of_batch(self):
        (self.workspace / 'a.py').write_text('X' * 100)
        runtime = Runtime(self.workspace)
        agent = self.agent(Client({'tool_calls': [call('a', 'read_file', path='a.py'), call('b', 'read_file', path='a.py')]}, {'content': 'done'}), runtime, output=60)
        agent.run('Task')
        contents = [i['result']['content'] for i in self.store.calls(agent.session_id)]
        self.assertIn('TRUNCATED', contents[0])
        self.assertEqual(contents[1], '~')
        self.assertEqual(runtime.quotas, [59, 1])
        self.assertEqual(self.spent(agent)['result_bytes'], 60)

    def test_tiny_truncated_error_remains_failed(self):
        agent = self.agent(Client({'tool_calls': [call('bad', 'read_file', path='../secret')]}, {'content': 'done'}), output=1)
        agent.run('Task')
        saved = self.store.calls(agent.session_id)[0]
        self.assertEqual(saved['state'], 'failed')
        self.assertEqual(saved['result']['content'], '~')

    def test_cached_write_costs_tool_and_output_but_no_second_effect(self):
        action = write()
        runtime = Runtime(self.workspace)
        agent = self.agent(Client({'tool_calls': [action]}, {'tool_calls': [action]}, {'content': 'done'}), runtime, tools=2, effects=1)
        agent.run('Task')
        self.assertEqual(runtime.executions, ['write_file'])
        self.assertEqual(self.spent(agent)['tool_calls'], 2)
        self.assertEqual(self.spent(agent)['side_effects'], 1)
        delivered = sum(len(m['content'].encode()) for m in self.store.get(agent.session_id).messages if m['role'] == 'tool')
        self.assertEqual(self.spent(agent)['result_bytes'], delivered)

    def test_refused_write_is_conservatively_charged(self):
        runtime = Runtime(self.workspace, mode='safe', approve=lambda request: False)
        agent = self.agent(Client({'tool_calls': [write()]}, {'content': 'refused'}), runtime)
        agent.run('Task')
        self.assertEqual(self.spent(agent)['side_effects'], 1)
        self.assertFalse((self.workspace / 'w.py').exists())

    def test_resume_finishes_reserved_batch_without_double_charge(self):
        first = self.agent(Client({'tool_calls': [write('a'), write('b')]}), on_event=self.pause_after_result)
        with self.assertRaises(KeyboardInterrupt):
            first.run('Task')
        before = self.spent(first)
        self.assertEqual(before['tool_calls'], 2)
        self.assertEqual(before['side_effects'], 2)
        self.store.db.close()
        self.store = SessionStore(self.path)
        resumed = self.agent(Client({'content': 'done'}), tools=999, effects=999)
        resumed.resume(first.session_id)
        self.assertEqual(self.spent(resumed)['tool_calls'], 2)
        self.assertEqual(self.spent(resumed)['side_effects'], 2)
        self.assertEqual(self.spent(resumed)['model_requests'], 2)
        self.assertEqual(resumed.budget['limits']['tool_calls'], 128)
        self.assertTrue((self.workspace / 'b.py').exists())

    def test_resume_waiting_approval_reuses_attempt_and_keeps_reservation(self):
        def defer(request):
            raise ApprovalDeferred('waiting')
        first = self.agent(Client({'tool_calls': [write()]}), Runtime(self.workspace, mode='safe', approve=defer))
        with self.assertRaises(ApprovalDeferred):
            first.run('Task')
        before = self.spent(first)
        self.assertEqual(before['result_bytes'], 1024)
        resumed = self.agent(Client({'content': 'done'}), Runtime(self.workspace, mode='safe', approve=lambda request: True))
        resumed.resume(first.session_id)
        self.assertEqual(self.spent(resumed)['tool_calls'], 1)
        self.assertEqual(self.spent(resumed)['side_effects'], 1)
        self.assertEqual(self.spent(resumed)['result_bytes'], before['result_bytes'])

    def test_plan_approval_and_resume_share_all_counters(self):
        responses = [{'tool_calls': [call('ls', 'list_files', path='.'), call('plan', 'submit_plan', **DEMO_PLAN)]}]
        first = self.agent(Client(*responses), plan_mode=True)
        with self.assertRaises(ApprovalDeferred):
            first.run('Task')
        before = self.spent(first)
        self.assertEqual(before['tool_calls'], 2)
        self.assertEqual(before['side_effects'], 0)
        self.store.decide_plan(first.session_id, True)
        self.assertEqual(self.spent(first), before)
        resumed = self.agent(Client({'tool_calls': [write()]}, {'content': 'done'}), approve_plan=lambda plan: True)
        resumed.resume(first.session_id)
        self.assertEqual(self.spent(resumed)['tool_calls'], 3)
        self.assertEqual(self.spent(resumed)['side_effects'], 1)
        self.assertEqual(self.spent(resumed)['model_requests'], 3)

    def hard_crash(self, when='after', tools=2, effects=2):
        script = '''import os,sys,json
from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.budget import BudgetLimits
from hw_ai_agent.session import SessionStore
from hw_ai_agent.tools import ToolRuntime
class Client:
 def complete(self,messages,tools):
  return {'tool_calls':[{'id':'w','function':{'name':'write_file','arguments':json.dumps({'path':'w.py','content':'effect'})}}]}
class Runtime(ToolRuntime):
 def execute(self,name,args):
  result=super().execute(name,args)
  os._exit(86)
with SessionStore(sys.argv[2]) as store:
 if sys.argv[3]=='before':
  original=store.start_call
  def start(*a,**kw):
   original(*a,**kw)
   os._exit(86)
  store.start_call=start
 runtime=Runtime(sys.argv[1],mode='auto') if sys.argv[3]=='after' else ToolRuntime(sys.argv[1],mode='auto')
 CodingAgent(Client(),runtime,store=store,budget_limits=BudgetLimits(12,int(sys.argv[4]),int(sys.argv[5]),1000000)).run('Task')
'''
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
        process = subprocess.run([sys.executable, '-c', script, str(self.workspace), str(self.path), when, str(tools), str(effects)], env=env, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 86, process.stderr.decode())
        return self.store.list_sessions()[0]['id']

    def test_hard_crash_between_debit_and_action_preserves_spending(self):
        identifier = self.hard_crash('before')
        before = self.store.budget(identifier)
        self.assertEqual(before['spent'], dict(model_requests=1, tool_calls=1, side_effects=1, result_bytes=1024))
        self.assertFalse((self.workspace / 'w.py').exists())
        with self.assertRaises(UncertainActionError):
            self.agent(Client()).resume(identifier)
        self.assertEqual(self.store.budget(identifier), before)

    def test_hard_crash_after_effect_retry_is_new_paid_attempt(self):
        identifier = self.hard_crash('after')
        before = self.store.budget(identifier)['spent']
        self.assertEqual((self.workspace / 'w.py').read_text(), 'effect')
        self.store.resolve(identifier, '1:0', 'retry')
        resumed = self.agent(Client({'content': 'done'}))
        resumed.resume(identifier)
        spent = self.store.budget(identifier)['spent']
        self.assertEqual(spent['tool_calls'], 2)
        self.assertEqual(spent['side_effects'], 2)
        self.assertGreater(spent['result_bytes'], before['result_bytes'])
        attempts = [e['attempt'] for e in self.store.get(identifier).events if e['event'] == 'tool_call']
        self.assertEqual(attempts, [1, 2])

    def test_retry_cannot_bypass_exhausted_effect_budget(self):
        identifier = self.hard_crash('after', effects=1)
        self.store.resolve(identifier, '1:0', 'retry')
        runtime = Runtime(self.workspace)
        with self.assertRaisesRegex(BudgetExceeded, 'side_effects'):
            self.agent(Client(), runtime).resume(identifier)
        self.assertEqual(runtime.executions, [])
        self.assertEqual(self.store.budget(identifier)['spent']['tool_calls'], 1)
        self.assertEqual(self.store.budget(identifier)['spent']['result_bytes'], 1024)

    def test_retry_cannot_bypass_exhausted_tool_budget(self):
        identifier = self.hard_crash('after', tools=1)
        self.store.resolve(identifier, '1:0', 'retry')
        runtime = Runtime(self.workspace)
        with self.assertRaisesRegex(BudgetExceeded, 'tool_calls'):
            self.agent(Client(), runtime).resume(identifier)
        self.assertEqual(runtime.executions, [])
        self.assertEqual(self.store.budget(identifier)['spent']['side_effects'], 1)

    def test_truncated_invalid_plan_result_does_not_save_a_plan(self):
        agent = self.agent(Client({'tool_calls': [call('plan', 'submit_plan', goal='incomplete')]}),
                           output=1, plan_mode=True, on_event=self.pause_after_result)
        with self.assertRaises(KeyboardInterrupt):
            agent.run('Task')
        self.assertEqual(self.store.calls(agent.session_id)[0]['state'], 'failed')
        self.assertEqual(self.store.plan(agent.session_id)['state'], 'planning')
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_applied_resolution_does_not_refund_lost_result_quota(self):
        identifier = self.hard_crash('after')
        before = self.store.budget(identifier)
        self.store.resolve(identifier, '1:0', 'applied', lambda item: True)
        self.assertEqual(self.store.budget(identifier), before)
        runtime = Runtime(self.workspace)
        self.agent(Client({'content': 'done'}), runtime).resume(identifier)
        self.assertEqual(runtime.executions, [])

    def test_checkpoint_failure_rolls_back_result_and_unused_quota_release(self):
        original = self.store._event
        def fail_result(identifier, event):
            if event['event'] == 'tool_result':
                raise SessionError('checkpoint failed')
            original(identifier, event)
        self.store._event = fail_result
        agent = self.agent(Client({'tool_calls': [write()]}))
        with self.assertRaises(SessionError):
            agent.run('Task')
        self.store._event = original
        self.assertTrue((self.workspace / 'w.py').exists())
        self.assertEqual(self.spent(agent)['result_bytes'], 1024)
        self.assertIsNone(self.store.calls(agent.session_id)[0]['result'])
        with self.assertRaises(UncertainActionError):
            self.agent(Client()).resume(agent.session_id)

    def test_batch_registration_and_debit_rollback_together(self):
        agent = self.agent(Client({'content': 'done'}))
        agent.run('Task')
        original = self.store._event
        def fail_debit(identifier, event):
            if event['event'] == 'budget_debit':
                raise SessionError('checkpoint failed')
            original(identifier, event)
        before = self.store.budget(agent.session_id)
        self.store._event = fail_debit
        with self.assertRaises(SessionError):
            self.store.assistant(agent.session_id, {'role': 'assistant', 'tool_calls': [write()]}, 2)
        self.store._event = original
        self.assertEqual(self.store.budget(agent.session_id), before)
        self.assertEqual(self.store.calls(agent.session_id), [])

    def test_corrupted_or_missing_budget_fails_before_model_call(self):
        for corruption in ['missing', 'negative', 'model mismatch', 'limits']:
            with self.subTest(corruption=corruption):
                first = self.agent(Client(RuntimeError('pause')))
                with self.assertRaises(RuntimeError):
                    first.run('Task')
                if corruption == 'missing':
                    self.store.db.execute('DELETE FROM budgets WHERE session_id=?', (first.session_id,))
                elif corruption == 'limits':
                    self.store.db.execute("UPDATE budgets SET limits='{}' WHERE session_id=?", (first.session_id,))
                else:
                    spent = self.spent(first)
                    spent['tool_calls' if corruption == 'negative' else 'model_requests'] = -1 if corruption == 'negative' else 0
                    self.store.db.execute('UPDATE budgets SET spent=? WHERE session_id=?', (json.dumps(spent), first.session_id))
                client = Client()
                with self.assertRaises(SessionError):
                    self.agent(client).resume(first.session_id)
                self.assertEqual(client.requests, [])

    def test_corrupted_reservation_cannot_authorize_tool(self):
        identifier = self.hard_crash('before')
        self.store.db.execute('UPDATE call_budgets SET output_quota=9999999 WHERE session_id=?', (identifier,))
        runtime = Runtime(self.workspace)
        with self.assertRaises(SessionError):
            self.agent(Client(), runtime).resume(identifier)
        self.assertEqual(runtime.executions, [])

    def test_legacy_v2_migration_preserves_history_and_reconstructs_spending(self):
        first = self.agent(Client({'tool_calls': [write(), call('r', 'read_file', path='w.py')]}), on_event=self.pause_after_result)
        with self.assertRaises(KeyboardInterrupt):
            first.run('Task')
        identifier = first.session_id
        before = self.store.get(identifier)
        self.store.db.execute('DROP TABLE call_budgets')
        self.store.db.execute('DROP TABLE budgets')
        self.store.db.execute('PRAGMA user_version=2')
        self.store.db.close()
        self.store = SessionStore(self.path)
        restored = self.store.get(identifier)
        self.assertEqual(restored.messages, before.messages)
        self.assertEqual(self.store.budget(identifier)['spent']['tool_calls'], 1)
        self.assertEqual(self.store.budget(identifier)['spent']['side_effects'], 1)
        self.agent(Client({'content': 'done'})).resume(identifier)
        self.assertEqual(self.store.budget(identifier)['spent']['tool_calls'], 2)
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 4)

    def test_legacy_version_marker_does_not_reset_existing_counters(self):
        first = self.agent(Client({'tool_calls': [write()]}, {'content': 'done'}))
        first.run('Task')
        before = self.store.budget(first.session_id)
        self.store.db.execute('PRAGMA user_version=1')
        self.store.db.close()
        self.store = SessionStore(self.path)
        self.assertEqual(self.store.budget(first.session_id), before)

    def test_completed_session_returns_cache_without_spending(self):
        first = self.agent(Client({'content': 'done'}))
        first.run('Task')
        before = self.store.budget(first.session_id)
        client = Client()
        self.assertEqual(self.agent(client).resume(first.session_id), 'done')
        self.assertEqual(client.requests, [])
        self.assertEqual(self.store.budget(first.session_id), before)


class RuntimeOutputBudgetTests(unittest.TestCase):
    def test_bash_receives_quota_and_bounds_whole_result(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = ToolRuntime(root, max_output_chars=10000)
            with runtime.output_budget(80), patch.object(runtime, '_sandbox', return_value=[sys.executable, '-c', "print('x'*1000)"]):
                output = runtime.bash('pwd')
            self.assertLessEqual(len(output.encode('utf-8')), 80)
            self.assertIn('TRUNCATED', output)
            self.assertIsNone(runtime.result_limit_bytes)

    def test_quota_restores_after_runtime_error(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = ToolRuntime(root)
            with self.assertRaises(ToolError), runtime.output_budget(10):
                runtime.execute('read_file', {'path': '../secret'})
            self.assertIsNone(runtime.result_limit_bytes)

    def test_list_output_marks_truncation(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = ToolRuntime(root)
            for i in range(10):
                Path(root, f'long_filename_{i}.py').touch()
            with runtime.output_budget(40):
                output = runtime.list_files()
            self.assertIn('TRUNCATED', output)
            self.assertLessEqual(len(output.encode()), 40)

    def test_standalone_agent_reserves_entire_batch(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = Runtime(root)
            agent = CodingAgent(Client({'tool_calls': [write('a'), write('b')]}), runtime,
                                budget_limits=BudgetLimits(tool_calls=1))
            with self.assertRaisesRegex(BudgetExceeded, 'tool_calls'):
                agent.run('Task')
            self.assertEqual(runtime.executions, [])

    def test_standalone_agent_has_same_output_limit(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'a.py').write_text('X' * 100)
            agent = CodingAgent(Client({'tool_calls': [call('r', 'read_file', path='a.py')]}, {'content': 'done'}), Runtime(root),
                                budget_limits=BudgetLimits(result_bytes=50))
            agent.run('Task')
            self.assertEqual(agent.budget['spent']['result_bytes'], 50)
            self.assertIn('TRUNCATED', agent.messages[-2]['content'])

    def test_invalid_limits_rejected_including_boolean_and_nan(self):
        for value in [-1, True, float('nan'), 1.5, '3']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                BudgetLimits(tool_calls=value)
        with self.assertRaises(ValueError):
            BudgetLimits(model_requests=0)
        with self.assertRaises(ValueError):
            truncate_result('x', 0)


class BudgetCLITests(unittest.TestCase):
    def test_budget_demo_is_offline_and_checks_all_dimensions(self):
        from hw_ai_agent.budget_demo import run_budget_demo
        report = run_budget_demo()
        self.assertTrue(report['success'], report['checks'])
        for name in DIMENSIONS:
            self.assertTrue(report['checks'][name + '_exhausted'])
        self.assertTrue(report['checks']['resume_preserves_counters'])
        self.assertTrue(report['checks']['crash_does_not_refund'])

    def test_budget_demo_cli_passes_without_loading_model_settings(self):
        from hw_ai_agent.cli import main
        with patch.object(sys, 'argv', ['hw-ai-agent', 'budget-demo']), \
             patch('hw_ai_agent.cli.load_env') as load, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 0)
            load.assert_not_called()
            self.assertIn('BUDGET DEMO PASS', output.getvalue())

    def test_run_zero_effects_has_no_unbudgeted_bash_preflight(self):
        from hw_ai_agent.cli import main
        from hw_ai_agent.config import Settings
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); workspace = root / 'workspace'; workspace.mkdir(); db = root / 'state.db'
            client = Client({'tool_calls': [write()]})
            client.settings = Settings('openrouter', 'example', 'https://example.org/v1', 'fake-key')
            runtime = Runtime(workspace, trash_dir=root / 'trash')
            with patch.object(sys, 'argv', ['hw-ai-agent', 'run', '--mode', 'auto', '--workspace', str(workspace),
                                         '--state-db', str(db), '--max-side-effects', '0', 'Fix files']), \
                 patch('hw_ai_agent.cli.load_env'), patch('hw_ai_agent.cli.Settings.from_env', return_value=client.settings), \
                 patch('hw_ai_agent.cli.OpenAICompatibleClient', return_value=client), \
                 patch('hw_ai_agent.cli.ToolRuntime', return_value=runtime), \
                 patch.object(runtime, 'bash') as bash, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as error:
                self.assertEqual(main(), 1)
                self.assertIn('BUDGET_EXHAUSTED', error.getvalue())
                bash.assert_not_called()
            self.assertEqual(runtime.executions, [])
            self.assertFalse((workspace / 'w.py').exists())

    def test_flags_on_start_commands_and_not_resume(self):
        from hw_ai_agent.cli import main
        for action in ['run', 'plan', 'demo', 'plan-demo', 'workflow-demo']:
            with self.subTest(action=action), patch.object(sys, 'argv', ['hw-ai-agent', action, '--help']), redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as result:
                    main()
                self.assertEqual(result.exception.code, 0)
                self.assertIn('--max-tool-calls', output.getvalue())
                self.assertIn('--max-side-effects', output.getvalue())
                self.assertIn('--max-result-bytes', output.getvalue())
        with patch.object(sys, 'argv', ['hw-ai-agent', 'resume', 'a'*32, '--max-tool-calls', '999']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main()
            self.assertEqual(result.exception.code, 2)

    def test_invalid_budget_fails_before_model_setup_or_workspace_creation(self):
        from hw_ai_agent.cli import main
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / 'workspace'
            with patch.object(sys, 'argv', ['hw-ai-agent', 'plan-demo', '--workspace', str(workspace), '--max-side-effects', '-1']), \
                 patch('hw_ai_agent.cli.load_env') as load, redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 1)
                load.assert_not_called()
            self.assertFalse(workspace.exists())

    def test_session_command_shows_saved_budget(self):
        from hw_ai_agent.cli import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); workspace = root / 'workspace'; workspace.mkdir(); db = root / 'state.db'
            with SessionStore(db) as store:
                agent = CodingAgent(Client({'content': 'done'}), Runtime(workspace), store=store)
                agent.run('Task')
                identifier = agent.session_id
            with patch.object(sys, 'argv', ['hw-ai-agent', 'session', identifier, '--state-db', str(db)]), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(), 0)
                data = json.loads(output.getvalue())
                self.assertEqual(data['budget']['spent']['model_requests'], 1)
                self.assertEqual(set(data['budget']['limits']), set(DIMENSIONS))


if __name__ == '__main__':
    unittest.main()
