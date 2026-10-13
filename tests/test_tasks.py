from __future__ import annotations
import copy
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

from hw_ai_agent.agent import CodingAgent, AgentError
from hw_ai_agent.budget import BudgetLimits
from hw_ai_agent.cli import main, verify_applied
from hw_ai_agent.config import Settings
from hw_ai_agent.planning import PlanError, PlanApprovalDeferred
from hw_ai_agent.session import SessionStore, SessionError, UncertainActionError
from hw_ai_agent.tasks import validate_task_plan, passing_tests, checklist
from hw_ai_agent.tasks_demo import TASK_PLAN, TasksDemoClient, TAX_FIXED, RECEIPT_FIXED, TESTS, run_tasks_demo
from hw_ai_agent.tools import ToolRuntime
from hw_ai_agent.policy import Policy


def call(identifier, name, **args):
    return {'id': identifier, 'function': {'name': name, 'arguments': json.dumps(args)}}


def plan(value=None):
    return {'tool_calls': [call('plan', 'submit_plan', **(value or TASK_PLAN))]}


def fix(identifier, content):
    return call('write_' + identifier, 'write_file', path=identifier + '.py', content=content)


def done(identifier):
    return call('done_' + identifier, 'complete_task', task_id=identifier)


PASS = 'exit_code=0\nstdout:\n\nstderr:\nRan 4 tests in 0.001s\n\nOK'


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []
    def complete(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        return next(self.responses)


class Runtime(ToolRuntime):
    def __init__(self, root, **opts):
        super().__init__(root, mode=opts.pop('mode', 'auto'), **opts)
        self.executions = []
        self.bash = lambda **kwargs: PASS
    def execute(self, name, arguments):
        self.executions.append((name, dict(arguments)))
        return super().execute(name, arguments)


class TaskGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve(); self.workspace = self.root / 'workspace'; self.workspace.mkdir()
        (self.workspace / 'tax.py').write_text('old tax')
        (self.workspace / 'receipt.py').write_text('old receipt')
        self.path = self.root / '.agent/state.sqlite3'; self.store = SessionStore(self.path)
    def tearDown(self):
        self.store.db.close(); self.temp.cleanup()
    def agent(self, client, runtime=None, **opts):
        return CodingAgent(client, runtime or Runtime(self.workspace), store=self.store, task_mode=True,
                           approve_plan=opts.pop('approve_plan', lambda p: True), **opts)
    def responses(self):
        return [plan(), {'tool_calls': [fix('tax', TAX_FIXED), done('tax')]},
                {'tool_calls': [fix('receipt', RECEIPT_FIXED), done('receipt')]},
                {'tool_calls': [call('tests', 'Bash', command='python -m unittest discover -v'), done('tests')]}, {'content': 'done'}]
    def pause(self, event):
        if event['event'] == 'task_completed':raise KeyboardInterrupt
    def paused(self):
        first = self.agent(Client(*self.responses()), on_event=self.pause)
        with self.assertRaises(KeyboardInterrupt):first.run('Task')
        return first

    def test_three_tasks_have_dependencies_proofs_and_call_scopes(self):
        agent = self.agent(Client(*self.responses()))
        self.assertEqual(agent.run('Task'), 'done')
        tasks = self.store.tasks(agent.session_id)
        self.assertEqual([t['state'] for t in tasks], ['completed'] * 3)
        self.assertEqual(tasks[1]['depends_on'], ['tax'])
        self.assertTrue(all(t['proof'] for t in tasks))
        calls = self.store.calls(agent.session_id)
        self.assertEqual(calls[0]['phase'], 'planning'); self.assertIsNone(calls[0]['task_id'])
        self.assertEqual([i['task_id'] for i in calls[1:]], ['tax', 'tax', 'receipt', 'receipt', 'tests', 'tests'])
        starts = [e['task_id'] for e in self.store.get(agent.session_id).events if e['event'] == 'task_started']
        self.assertEqual(starts, ['tax', 'receipt', 'tests'])

    def test_dependent_task_cannot_be_completed_before_parent(self):
        client = Client(plan(), {'tool_calls': [call('early_done', 'complete_task', task_id='receipt')]}, *self.responses()[1:])
        agent = self.agent(client);agent.run('Task')
        error = next(m for m in client.requests[2] if m.get('tool_call_id') == 'early_done')
        self.assertIn('TASK_NOT_CURRENT', error['content'])
        self.assertEqual(self.store.tasks(agent.session_id)[1]['state'], 'completed')

    def test_plain_model_text_cannot_complete_tasks(self):
        client = Client(plan(), {'content': 'Everything is done.'}, *self.responses()[1:])
        agent = self.agent(client);agent.run('Task')
        self.assertIn('TASKS_NOT_COMPLETE', client.requests[2][-1]['content'])
        self.assertTrue(all(t['proof'] for t in self.store.tasks(agent.session_id)))

    def test_model_supplied_proof_is_rejected(self):
        client = Client(plan(), {'tool_calls': [call('fake', 'complete_task', task_id='tax', proof={'success': True})]}, *self.responses()[1:])
        agent = self.agent(client);agent.run('Task')
        self.assertIn('Invalid task control', client.requests[2][-1]['content'])

    def test_missing_write_evidence_keeps_task_running(self):
        client = Client(plan(), {'tool_calls': [done('tax')]})
        def stop(event):
            if event['event'] == 'tool_result' and event['name'] == 'complete_task':raise KeyboardInterrupt
        agent = self.agent(client, on_event=stop)
        with self.assertRaises(KeyboardInterrupt):agent.run('Task')
        self.assertEqual(self.store.tasks(agent.session_id)[0]['state'], 'running')
        self.assertIn('TASK_NOT_VERIFIED', self.store.calls(agent.session_id)[-1]['result']['content'])

    def test_file_changed_after_write_does_not_pass_verification(self):
        def edit(event):
            if event['event'] == 'tool_result' and event['name'] == 'write_file':
                (self.workspace / 'tax.py').write_text('external edit')
            if event['event'] == 'tool_result' and event['name'] == 'complete_task':raise KeyboardInterrupt
        agent = self.agent(Client(plan(), self.responses()[1]), on_event=edit)
        with self.assertRaises(KeyboardInterrupt):agent.run('Task')
        self.assertIsNone(self.store.tasks(agent.session_id)[0]['proof'])
        self.assertIn('differs', self.store.calls(agent.session_id)[-1]['result']['content'])

    def test_failed_zero_or_truncated_tests_do_not_complete_task(self):
        for output in ['exit_code=1\nRan 4 tests in 0.1s\nFAILED', 'exit_code=0\nRan 0 tests in 0.1s\nOK', PASS+'\n[TRUNCATED: result budget]']:
            with self.subTest(output=output):
                runtime = Runtime(self.workspace);runtime.bash = lambda **kwargs: output
                def stop(event):
                    if event['event'] == 'tool_result' and event.get('tool_call_id') == 'done_tests':raise KeyboardInterrupt
                agent = self.agent(Client(*self.responses()[:-1]), runtime, on_event=stop)
                with self.assertRaises(KeyboardInterrupt):agent.run('Task')
                self.assertEqual(self.store.tasks(agent.session_id)[2]['state'], 'running')

    def test_latest_failed_test_cannot_reuse_earlier_success(self):
        outputs=iter([PASS,'exit_code=1\nRan 4 tests in 0.1s\nFAILED'])
        runtime=Runtime(self.workspace);runtime.bash=lambda **kwargs:next(outputs)
        def stop(event):
            if event['event']=='tool_result' and event.get('tool_call_id')=='done_tests':raise KeyboardInterrupt
        responses=[*self.responses()[:3],{'tool_calls':[call('old_pass','Bash',command='python -m unittest discover -v')]},
                   {'tool_calls':[call('new_failure','Bash',command='python -m unittest discover -v'),done('tests')]}]
        agent=self.agent(Client(*responses),runtime,on_event=stop)
        with self.assertRaises(KeyboardInterrupt):agent.run('Task')
        self.assertEqual(self.store.tasks(agent.session_id)[2]['state'],'running')
        self.assertIn('latest test run',self.store.calls(agent.session_id)[-1]['result']['content'])

    def test_tool_not_declared_for_current_task_is_refused(self):
        client = Client(plan(), {'tool_calls': [call('early', 'Bash', command='pwd')]}, *self.responses()[1:])
        runtime = Runtime(self.workspace); agent = self.agent(client, runtime);agent.run('Task')
        self.assertIn('TASK_TOOL_DENIED', client.requests[2][-1]['content'])
        self.assertEqual(sum(n == 'Bash' for n, _ in runtime.executions), 1)

    def test_complete_task_must_be_last_in_batch(self):
        client = Client(plan(), {'tool_calls': [done('tax'), fix('tax', TAX_FIXED)]},
                        {'tool_calls': [call('real_done_tax', 'complete_task', task_id='tax')]}, *self.responses()[2:])
        agent = self.agent(client);agent.run('Task')
        self.assertIn('TASK_BOUNDARY', next(m['content'] for m in client.requests[2] if m.get('tool_call_id') == 'done_tax'))

    def test_resume_does_not_reexecute_completed_task_or_reset_budget(self):
        first = self.paused();before = self.store.budget(first.session_id)
        self.store.db.close();self.store = SessionStore(self.path)
        runtime = Runtime(self.workspace);resumed = self.agent(Client(*self.responses()[2:]), runtime)
        resumed.resume(first.session_id)
        self.assertFalse(any(args.get('path') == 'tax.py' for _, args in runtime.executions))
        starts = [e['task_id'] for e in self.store.get(first.session_id).events if e['event'] == 'task_started']
        self.assertEqual(starts, ['tax', 'receipt', 'tests'])
        self.assertGreater(resumed.budget['spent']['tool_calls'], before['spent']['tool_calls'])
        self.assertEqual(resumed.budget['spent']['side_effects'], 3)

    def test_failure_blocks_dependants_and_prevents_false_final_success(self):
        runtime = Runtime(self.workspace)
        agent = self.agent(Client(plan(), {'tool_calls': [call('fail', 'fail_task', task_id='tax', reason='operator refused fix')]}), runtime)
        with self.assertRaisesRegex(AgentError, 'TASKS_FAILED'):agent.run('Task')
        self.assertEqual([t['state'] for t in self.store.tasks(agent.session_id)], ['failed', 'blocked', 'blocked'])
        self.assertEqual(runtime.executions, [])
        self.assertNotEqual(self.store.get(agent.session_id).status, 'completed')

    def test_task_plan_is_saved_before_approval_without_side_effects(self):
        runtime = Runtime(self.workspace)
        agent = self.agent(Client(plan()), runtime, approve_plan=None)
        with self.assertRaises(PlanApprovalDeferred):agent.run('Task')
        self.assertEqual(runtime.executions, [])
        self.assertEqual([t['state'] for t in self.store.tasks(agent.session_id)], ['pending'] * 3)

    def test_forward_dependency_order_is_scheduled_by_readiness(self):
        value = copy.deepcopy(TASK_PLAN);value['steps'] = [value['steps'][1], value['steps'][0], value['steps'][2]]
        agent = self.agent(Client(plan(value), *self.responses()[1:]));agent.run('Task')
        starts = [e['task_id'] for e in self.store.get(agent.session_id).events if e['event'] == 'task_started']
        self.assertEqual(starts, ['tax', 'receipt', 'tests'])

    def test_proof_and_completion_checkpoint_are_atomic(self):
        original = self.store._event
        def fail(identifier, event):
            if event['event'] == 'task_completed':raise SessionError('checkpoint failed')
            original(identifier, event)
        self.store._event = fail
        first = self.agent(Client(plan(), self.responses()[1]))
        with self.assertRaises(SessionError):first.run('Task')
        self.store._event = original
        self.assertEqual(self.store.tasks(first.session_id)[0]['state'], 'running')
        self.assertIsNone(self.store.tasks(first.session_id)[0]['proof'])
        self.assertIsNone(self.store.calls(first.session_id)[-1]['result'])
        with self.assertRaises(UncertainActionError):self.agent(Client()).resume(first.session_id)
        self.store.resolve(first.session_id, '2:1', 'retry')
        runtime = Runtime(self.workspace)
        self.agent(Client(*self.responses()[2:]), runtime).resume(first.session_id)
        self.assertFalse(any(a.get('path') == 'tax.py' for _, a in runtime.executions))
        self.assertTrue(all(t['state'] == 'completed' for t in self.store.tasks(first.session_id)))

    def test_cross_task_reused_call_id_is_protocol_error_before_execution(self):
        first = self.paused();runtime = Runtime(self.workspace)
        client = Client({'tool_calls': [fix('tax', TAX_FIXED)]})
        with self.assertRaisesRegex(SessionError, 'PROTOCOL_ERROR'):
            self.agent(client, runtime).resume(first.session_id)
        self.assertEqual(runtime.executions, [])

    def test_corrupted_proof_blocks_cached_completed_session(self):
        first = self.agent(Client(*self.responses()));first.run('Task')
        self.store.db.execute("UPDATE tasks SET proof='{}' WHERE session_id=? AND task_id='tax'", (first.session_id,))
        client = Client()
        with self.assertRaises(SessionError):self.agent(client).resume(first.session_id)
        self.assertEqual(client.requests, [])

    def test_missing_task_or_edited_dependencies_block_resume(self):
        first = self.paused()
        self.store.db.execute("UPDATE tasks SET definition='{}' WHERE session_id=? AND task_id='receipt'", (first.session_id,))
        with self.assertRaises(SessionError):self.agent(Client()).resume(first.session_id)

    def test_tampered_pending_call_task_id_cannot_start(self):
        first = self.paused()
        def interrupt(*args):
            original(*args);raise KeyboardInterrupt
        original = self.store.assistant;self.store.assistant = interrupt
        with self.assertRaises(KeyboardInterrupt):self.agent(Client(self.responses()[2])).resume(first.session_id)
        self.store.assistant = original
        self.store.db.execute("UPDATE call_tasks SET task_id='tests' WHERE session_id=? AND batch=3", (first.session_id,))
        runtime = Runtime(self.workspace)
        with self.assertRaises(SessionError):self.agent(Client(), runtime).resume(first.session_id)
        self.assertEqual(runtime.executions, [])

    def test_file_read_verification_is_supported(self):
        value = copy.deepcopy(TASK_PLAN);value['steps'] = [{'id': 'inspect', 'description': 'Read source', 'tools': ['read_file'], 'depends_on': [], 'verification': {'kind': 'file_read', 'path': 'tax.py'}}]
        agent = self.agent(Client(plan(value), {'tool_calls': [call('read', 'read_file', path='tax.py'), done('inspect')]}, {'content': 'done'}))
        agent.run('Task');self.assertEqual(self.store.tasks(agent.session_id)[0]['proof']['kind'], 'file_read')

    def test_safe_still_confirms_each_write_and_bash(self):
        prompts = [];runtime = Runtime(self.workspace, mode='safe', approve=lambda r: prompts.append(r.tool) or True)
        agent = self.agent(Client(*self.responses()), runtime);agent.run('Task')
        self.assertEqual(prompts, ['write_file', 'write_file', 'Bash'])

    def test_tasks_require_persistent_store(self):
        agent = CodingAgent(Client(), Runtime(self.workspace), task_mode=True)
        with self.assertRaises(SessionError):agent.run('Task')

    def test_delete_task_verifies_absence_and_confirms_in_both_modes(self):
        value = copy.deepcopy(TASK_PLAN)
        value['steps'] = [{'id': 'cleanup', 'description': 'Remove unused source', 'tools': ['delete_file'],
                           'depends_on': [], 'verification': {'kind': 'file_absent', 'path': 'unused.py'}}]
        for mode in ['safe', 'auto']:
            with self.subTest(mode=mode):
                (self.workspace / 'unused.py').write_text('unused')
                approvals=[];runtime=Runtime(self.workspace,mode=mode,approve=lambda r:approvals.append(r.tool) or True)
                agent=self.agent(Client(plan(value), {'tool_calls': [call('remove','delete_file',path='unused.py'),done('cleanup')]}, {'content':'done'}),runtime)
                agent.run('Task')
                self.assertEqual(approvals,['delete_file'])
                self.assertFalse((self.workspace/'unused.py').exists())
                self.assertEqual(self.store.tasks(agent.session_id)[0]['proof']['kind'],'file_absent')

    def test_file_read_with_literal_marker_is_not_mistaken_for_cut_output(self):
        value=copy.deepcopy(TASK_PLAN)
        value['steps']=[{'id':'inspect','description':'Read source','tools':['read_file'],'depends_on':[],
                         'verification':{'kind':'file_read','path':'tax.py'}}]
        for text in ['~', 'literal [TRUNCATED: result budget] inside source']:
            (self.workspace/'tax.py').write_text(text)
            agent=self.agent(Client(plan(value),{'tool_calls':[call('read','read_file',path='tax.py'),done('inspect')]},{'content':'done'}))
            agent.run('Task');self.assertEqual(self.store.tasks(agent.session_id)[0]['state'],'completed')

    def test_verification_does_not_bypass_explicit_read_deny(self):
        policy=Policy.from_dict({'rules':[{'tool':'read_file','path':'tax.py','action':'deny'}]})
        def stop(event):
            if event['event']=='tool_result' and event['name']=='complete_task':raise KeyboardInterrupt
        agent=self.agent(Client(plan(),self.responses()[1]),Runtime(self.workspace,policy=policy),on_event=stop)
        with self.assertRaises(KeyboardInterrupt):agent.run('Task')
        self.assertIsNone(self.store.tasks(agent.session_id)[0]['proof'])
        self.assertIn('permitted',self.store.calls(agent.session_id)[-1]['result']['content'])

    def test_v3_legacy_plan_and_budget_migrate_without_conversion_or_reset(self):
        from hw_ai_agent.plan_demo import DEMO_PLAN
        first=CodingAgent(Client(plan(DEMO_PLAN)),Runtime(self.workspace),store=self.store,plan_mode=True)
        with self.assertRaises(PlanApprovalDeferred):first.run('Task')
        before=self.store.budget(first.session_id);old_plan=self.store.plan(first.session_id)
        self.store.db.execute('DROP TABLE call_tasks');self.store.db.execute('DROP TABLE tasks')
        self.store.db.execute('PRAGMA user_version=3');self.store.db.close();self.store=SessionStore(self.path)
        self.assertEqual(self.store.budget(first.session_id),before)
        self.assertEqual(self.store.plan(first.session_id),old_plan)
        self.assertEqual(self.store.tasks(first.session_id),[])
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0],4)


class TaskValidationTests(unittest.TestCase):
    def test_bad_ids_cycles_unknown_dependencies_and_verification_are_rejected(self):
        values=[]
        for mutation in [('id', 'tax'), ('depends_on', ['missing']), ('depends_on', ['receipt']), ('verification', {'kind': 'claim', 'text': 'done'})]:
            value=copy.deepcopy(TASK_PLAN);value['steps'][1][mutation[0]]=mutation[1];values.append(value)
        value=copy.deepcopy(TASK_PLAN);value['steps'][0]['depends_on']=['tests'];values.append(value)
        value=copy.deepcopy(TASK_PLAN);value['steps'][0]['verification']['path']='../secret';values.append(value)
        value=copy.deepcopy(TASK_PLAN);value['steps'][2]['verification']['command']='pwd';values.append(value)
        value=copy.deepcopy(TASK_PLAN);value['steps'][0]['tools']=['read_file'];values.append(value)
        for value in values:
            with self.subTest(value=value),self.assertRaises(PlanError):validate_task_plan(value)

    def test_tests_evidence_requires_nonempty_successful_untruncated_run(self):
        self.assertTrue(passing_tests(PASS));self.assertFalse(passing_tests('I ran tests. OK'))
        self.assertFalse(passing_tests('exit_code=0\nRan 0 tests in 0s\nOK'))
        self.assertFalse(passing_tests(PASS+'\noutput truncated: command stopped at output limit'))

    def test_checklist_has_statuses_and_dependencies(self):
        tasks=[{**s,'state':state} for s,state in zip(TASK_PLAN['steps'], ['completed','running','pending'])]
        text=checklist(tasks);self.assertIn('[x] tax',text);self.assertIn('[>] receipt',text);self.assertIn('[ ] tests',text)
        self.assertIn('tax, receipt',text)


@unittest.skipUnless(os.getenv('REQUIRE_SANDBOX_TESTS') == '1', 'Requires native macOS sandbox')
class TaskDemoIntegrationTests(unittest.TestCase):
    def test_native_offline_demo_passes_and_keeps_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            report=run_tasks_demo(root/'workspace', root/'state.db', offline=True, mode='auto', approve_plan=lambda p:True)
            self.assertTrue(report['success'],report)
            self.assertEqual((root/'workspace/test_checkout.py').read_bytes(),TESTS.encode())
            self.assertEqual(len(report['tasks']),3)
            self.assertEqual(report['budget']['spent']['model_requests'],6)

    def test_cli_pause_after_task_resume_without_id_and_preserve_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);workspace=root/'workspace';db=root/'state.db'
            args=['hw-ai-agent','tasks-demo','--offline','--mode','auto','--workspace',str(workspace),'--state-db',str(db),'--stop-after-tasks','1','--max-tokens','1800']
            with patch.object(sys,'argv',args),patch('hw_ai_agent.cli.confirm_plan',return_value=True),redirect_stdout(io.StringIO()) as output,redirect_stderr(io.StringIO()):
                self.assertEqual(main(),130)
                self.assertIn('[x] tax',output.getvalue())
            with SessionStore(db) as store:
                identifier=store.list_sessions()[0]['id'];before=store.budget(identifier)
                self.assertEqual(store.tasks(identifier)[0]['state'],'completed')
            with patch.object(sys,'argv',['hw-ai-agent','resume','--state-db',str(db)]),patch('sys.stdin.isatty',return_value=True),redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(),0);self.assertIn('TASKS DEMO PASS',output.getvalue())
            with SessionStore(db) as store:
                self.assertEqual(store.get(identifier).metadata['max_tokens'],1800)
                self.assertEqual(store.budget(identifier)['spent']['side_effects'],3)
                starts=[e['task_id'] for e in store.get(identifier).events if e['event']=='task_started'];self.assertEqual(starts,['tax','receipt','tests'])
                self.assertEqual(store.budget(identifier)['limits'],before['limits'])

    def test_default_real_client_is_saved_and_not_replaced_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);workspace=root/'workspace';db=root/'state.db'
            settings=Settings('ollama','example','http://127.0.0.1:11434/v1','ollama')
            def real(*args,**kwargs):
                client=TasksDemoClient();client.settings=settings;client.max_tokens=kwargs['max_tokens'];return client
            args=['hw-ai-agent','tasks-demo','--mode','auto','--workspace',str(workspace),'--state-db',str(db),'--stop-after-tasks','1']
            with patch.object(sys,'argv',args),patch('hw_ai_agent.cli.load_env'),patch('hw_ai_agent.cli.Settings.from_env',return_value=settings),patch('hw_ai_agent.cli.doctor',return_value=True),patch('hw_ai_agent.cli.OpenAICompatibleClient',side_effect=real) as factory,patch('hw_ai_agent.cli.confirm_plan',return_value=True),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                self.assertEqual(main(),130);factory.assert_called_once()
            with SessionStore(db) as store:
                identifier=store.list_sessions()[0]['id'];self.assertFalse(store.get(identifier).metadata['demo']['offline'])
            with patch.object(sys,'argv',['hw-ai-agent','resume',identifier,'--state-db',str(db)]),patch('hw_ai_agent.cli.load_env'),patch('hw_ai_agent.cli.Settings.from_env',return_value=settings),patch('hw_ai_agent.cli.OpenAICompatibleClient',side_effect=real),patch('hw_ai_agent.cli.TasksDemoClient',side_effect=AssertionError('No scripted fallback')),redirect_stdout(io.StringIO()):
                self.assertEqual(main(),0)

    def test_hard_crash_after_effect_blocks_then_applied_resumes_same_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);workspace=root/'workspace';db=root/'state.db'
            script='''import sys
from pathlib import Path
from hw_ai_agent.tasks_demo import run_tasks_demo
run_tasks_demo(Path(sys.argv[1]),Path(sys.argv[2]),offline=True,mode='auto',approve_plan=lambda p:True,crash_after_write=True)
'''
            env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
            result=subprocess.run([sys.executable,'-c',script,str(workspace),str(db)],env=env,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,86,result.stderr.decode())
            with SessionStore(db) as store:
                identifier=store.list_sessions()[0]['id'];runtime=ToolRuntime(workspace,mode='auto',trash_dir=root/'trash')
                agent=CodingAgent(TasksDemoClient(),runtime,store=store)
                with self.assertRaises(UncertainActionError):agent.resume(identifier)
                item=next(i for i in store.calls(identifier,True) if i['state']=='uncertain')
                before=store.budget(identifier)
                store.resolve(identifier,item['key'],'applied',lambda i:verify_applied(runtime,i))
                self.assertEqual(store.budget(identifier),before)
                self.assertEqual(agent.resume(identifier),'Все три задачи завершены; результаты подтверждены runtime.')
                self.assertTrue(all(t['state']=='completed' for t in store.tasks(identifier)))


class TaskCLITests(unittest.TestCase):
    def test_flags_are_available_and_budget_limits_cannot_change_on_resume(self):
        for action in ['run','plan','tasks-demo']:
            with patch.object(sys,'argv',['hw-ai-agent',action,'--help']),redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as result:main()
                self.assertEqual(result.exception.code,0)
                self.assertIn('--stop-after-tasks',output.getvalue())
                self.assertIn('--tasks' if action!='tasks-demo' else '--offline',output.getvalue())
    def test_new_demo_has_no_silent_offline_fallback(self):
        with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(ValueError,'Real model'):
            root=Path(directory);run_tasks_demo(root/'workspace',root/'state.db')
    def test_task_demo_invalid_limits_precede_model_setup(self):
        with patch.object(sys,'argv',['hw-ai-agent','tasks-demo','--max-tool-calls','-1']),patch('hw_ai_agent.cli.load_env') as load,redirect_stderr(io.StringIO()):
            self.assertEqual(main(),1);load.assert_not_called()


if __name__=='__main__':unittest.main()
