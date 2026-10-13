from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

from hw_ai_agent.agent import CodingAgent, AgentError
from hw_ai_agent.cli import main, verify_applied
from hw_ai_agent.plan_demo import DEMO_PLAN, PlanDemoClient, run_plan_demo
from hw_ai_agent.config import Settings
from hw_ai_agent.planning import PlanApprovalDeferred, PlanError, validate_plan
from hw_ai_agent.session import SessionStore, SessionError, PlanRejectedError, UncertainActionError
from hw_ai_agent.terminal import confirm_plan, confirm_conflict, choose_session
from hw_ai_agent.tools import ToolRuntime, ToolError
from hw_ai_agent.workflow_demo import call


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []
        self.schemas = []

    def complete(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        self.schemas.append(copy.deepcopy(tools))
        return next(self.responses)


def submit(identifier='plan', value=None):
    return call(identifier, 'submit_plan', DEMO_PLAN if value is None else value)


def write(identifier='write'):
    return call(identifier, 'write_file', {'path': 'order.py', 'content': 'fixed'})


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        (self.workspace / 'order.py').write_text('original')
        self.store = SessionStore(self.root / '.agent/state.sqlite3')

    def tearDown(self):
        self.store.db.close()
        self.temporary.cleanup()

    def runtime(self, mode='safe', approve=None):
        runtime = ToolRuntime(self.workspace, mode=mode, approve=approve)
        runtime.bash = Mock(return_value='exit_code=0\nOK')
        return runtime

    def test_planning_blocks_every_side_effect_even_with_operator_approval(self):
        for mode in ('safe', 'auto'):
            with self.subTest(mode=mode):
                approve = Mock(return_value=True)
                runtime = self.runtime(mode, approve)
                forbidden = [write('early'), call('delete', 'delete_file', {'path': 'order.py'}),
                             call('cmd', 'Bash', {'command': 'pwd'}),
                             call('read', 'read_file', {'path': 'order.py'})]
                def decide(plan):
                    self.assertEqual((self.workspace / 'order.py').read_text(), 'original')
                    runtime.bash.assert_not_called()
                    approve.assert_not_called()
                    return False
                agent = CodingAgent(Client({'tool_calls': forbidden}, {'tool_calls': [submit()]}),
                                    runtime, store=self.store, plan_mode=True, approve_plan=decide)
                with self.assertRaises(PlanRejectedError):
                    agent.run('Task')
                results = [m['content'] for m in agent.messages if m['role'] == 'tool']
                self.assertEqual(sum('PLAN_MODE:' in r for r in results), 3)
                self.assertEqual(self.store.get(agent.session_id).status, 'aborted')

    def test_runtime_direct_write_and_bash_are_also_blocked(self):
        runtime = ToolRuntime(self.workspace, approve=lambda request: True)
        runtime.phase = 'planning'
        for action in (lambda: runtime.write_file('order.py', 'bad'), lambda: runtime.bash('pwd')):
            with self.assertRaisesRegex(ToolError, 'PLAN_MODE'):
                action()
        self.assertEqual((self.workspace / 'order.py').read_text(), 'original')

    def test_plan_schemas_expose_only_reading_and_submit(self):
        client = Client({'tool_calls': [submit()]})
        agent = CodingAgent(client, self.runtime(), store=self.store, plan_mode=True)
        with self.assertRaises(PlanApprovalDeferred):
            agent.run('Task')
        self.assertEqual({t['function']['name'] for t in client.schemas[0]}, {'read_file', 'list_files', 'submit_plan'})

    def test_approval_keeps_safe_and_auto_permissions(self):
        for mode in ('safe', 'auto'):
            with self.subTest(mode=mode):
                approve = Mock(return_value=True)
                runtime = self.runtime(mode, approve)
                client = Client({'tool_calls': [submit()]}, {'tool_calls': [write(), call('cmd', 'Bash', {'command': 'pwd'})]},
                                {'content': 'done'})
                complete = client.complete
                def checked_complete(messages, tools):
                    if client.requests:
                        self.assertEqual(self.store.get(agent.session_id).status, 'running')
                    return complete(messages, tools)
                client.complete = checked_complete
                decisions = []
                def decide(plan):
                    saved = self.store.plan(agent.session_id)
                    self.assertEqual(saved['state'], 'awaiting_approval')
                    self.assertEqual(saved['data'], DEMO_PLAN)
                    self.assertEqual((self.workspace / 'order.py').read_text(), 'original')
                    self.assertEqual(self.store.calls(agent.session_id, True), [])
                    decisions.append(plan)
                    return True
                agent = CodingAgent(client, runtime, store=self.store, plan_mode=True, approve_plan=decide)
                self.assertEqual(agent.run('Task'), 'done')
                self.assertEqual(len(decisions), 1)
                self.assertEqual(approve.call_count, 2 if mode == 'safe' else 0)
                self.assertEqual(self.store.get(agent.session_id).rounds, 3)
                self.assertEqual(self.store.plan(agent.session_id)['state'], 'approved')
                self.assertEqual((self.workspace / 'order.py').read_text(), 'fixed')
                (self.workspace / 'order.py').write_text('original')

    def test_plan_rejection_is_terminal_without_execution(self):
        agent = CodingAgent(Client({'tool_calls': [submit()]}), self.runtime(), store=self.store,
                            plan_mode=True, approve_plan=lambda plan: False)
        with self.assertRaises(PlanRejectedError):
            agent.run('Task')
        with self.assertRaisesRegex(SessionError, 'terminated'):
            CodingAgent(Client(), self.runtime(), store=self.store).resume(agent.session_id)
        self.assertEqual((self.workspace / 'order.py').read_text(), 'original')
        self.assertEqual(self.store.plan(agent.session_id)['state'], 'rejected')

    def test_no_implicit_approval_when_callback_missing_invalid_or_broken(self):
        for decide in (None, lambda plan: 1, Mock(side_effect=RuntimeError('broken'))):
            agent = CodingAgent(Client({'tool_calls': [submit()]}), self.runtime('auto'), store=self.store,
                                plan_mode=True, approve_plan=decide)
            with self.assertRaises(PlanApprovalDeferred):
                agent.run('Task')
            self.assertEqual(self.store.get(agent.session_id).status, 'waiting_plan_approval')
            self.assertEqual(self.store.plan(agent.session_id)['state'], 'awaiting_approval')
            self.assertEqual((self.workspace / 'order.py').read_text(), 'original')

    def test_resume_reopened_database_asks_plan_before_any_execution(self):
        first = CodingAgent(Client({'tool_calls': [submit()]}), self.runtime('auto'), store=self.store, plan_mode=True)
        with self.assertRaises(PlanApprovalDeferred):
            first.run('Task')
        identifier = first.session_id
        self.store.db.close()
        self.store = SessionStore(self.root / '.agent/state.sqlite3')
        decide = Mock(return_value=True)
        client = Client({'tool_calls': [write()]}, {'content': 'done'})
        self.assertEqual(CodingAgent(client, self.runtime('auto'), store=self.store, approve_plan=decide).resume(identifier), 'done')
        decide.assert_called_once()
        self.assertEqual(self.store.get(identifier).rounds, 3)
        self.assertEqual(len([c for c in self.store.calls(identifier) if c['call']['function']['name'] == 'submit_plan']), 1)

    def test_interrupt_at_plan_question_keeps_saved_plan(self):
        first = CodingAgent(Client({'tool_calls': [submit()]}), self.runtime(), store=self.store,
                            plan_mode=True, approve_plan=Mock(side_effect=KeyboardInterrupt))
        with self.assertRaises(KeyboardInterrupt):
            first.run('Task')
        self.assertEqual(self.store.plan(first.session_id)['state'], 'awaiting_approval')
        self.assertEqual(self.store.calls(first.session_id, True), [])
        with self.assertRaises(PlanApprovalDeferred):
            CodingAgent(Client(), self.runtime(), store=self.store).resume(first.session_id)

    def test_invalid_plan_returns_tool_error_and_can_be_corrected(self):
        client = Client({'tool_calls': [submit('bad', {'goal': 'missing fields'})]}, {'tool_calls': [submit('good')]})
        agent = CodingAgent(client, self.runtime(), store=self.store, plan_mode=True)
        with self.assertRaises(PlanApprovalDeferred):
            agent.run('Task')
        self.assertTrue(client.requests[1][-1]['content'].startswith('ERROR:'))
        self.assertEqual(self.store.plan(agent.session_id)['data'], DEMO_PLAN)

    def test_text_answer_cannot_complete_planning(self):
        agent = CodingAgent(Client({'content': 'I approve my own plan'}, {'tool_calls': [submit()]}),
                            self.runtime(), store=self.store, plan_mode=True)
        with self.assertRaises(PlanApprovalDeferred):
            agent.run('Task')
        self.assertEqual(self.store.get(agent.session_id).status, 'waiting_plan_approval')

    def test_unsafe_calls_after_submit_in_same_batch_are_denied_before_approval(self):
        client = Client({'tool_calls': [submit(), write('early')]}, {'content': 'done'})
        agent = CodingAgent(client, self.runtime('auto'), store=self.store, plan_mode=True, approve_plan=lambda plan: True)
        agent.run('Task')
        self.assertEqual((self.workspace / 'order.py').read_text(), 'original')
        self.assertIn('PLAN_MODE', self.store.calls(agent.session_id)[1]['result']['content'])

    def test_approval_callback_cannot_mutate_the_saved_plan(self):
        def decide(plan):
            plan['goal'] = 'other goal'
            return True
        agent = CodingAgent(Client({'tool_calls': [submit()]}, {'content': 'done'}), self.runtime(),
                            store=self.store, plan_mode=True, approve_plan=decide)
        agent.run('Task')
        self.assertEqual(self.store.plan(agent.session_id)['data'], DEMO_PLAN)

    def test_external_approval_and_rejection_commands(self):
        for command in ('approve-plan', 'reject-plan'):
            first = CodingAgent(Client({'tool_calls': [submit()]}), self.runtime(), store=self.store, plan_mode=True)
            with self.assertRaises(PlanApprovalDeferred):
                first.run('Task')
            with patch('sys.argv', ['hw-ai-agent', command, first.session_id, '--state-db', str(self.store.path)]), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(self.store.plan(first.session_id)['state'], 'approved' if command == 'approve-plan' else 'rejected')

    def test_approval_cannot_bypass_uncertain_or_pending_plan_calls(self):
        agent = CodingAgent(Client(), self.runtime(), store=self.store, plan_mode=True)
        identifier = self.store.create(agent._binding(), [{'role': 'user', 'content': 'Task'}], 12, 120000)
        with self.assertRaises(SessionError):
            self.store.decide_plan(identifier, True)

    def test_plan_checkpoint_is_atomic_with_tool_result(self):
        first = CodingAgent(Client({'tool_calls': [submit()]}), self.runtime(), store=self.store, plan_mode=True)
        original = self.store._event
        def fail(identifier, event):
            if event['event'] == 'plan_saved':
                raise SessionError('crash during checkpoint')
            return original(identifier, event)
        with patch.object(self.store, '_event', side_effect=fail), self.assertRaises(SessionError):
            first.run('Task')
        self.assertEqual(self.store.plan(first.session_id)['state'], 'planning')
        self.assertEqual(self.store.calls(first.session_id)[0]['state'], 'running')
        self.store.resolve(first.session_id, '1:0', 'retry')
        with self.assertRaises(PlanApprovalDeferred):
            CodingAgent(Client(), self.runtime(), store=self.store).resume(first.session_id)
        self.assertEqual(self.store.plan(first.session_id)['data'], DEMO_PLAN)

    def test_plan_mode_requires_persistence(self):
        with self.assertRaises(SessionError):
            CodingAgent(Client(), self.runtime(), plan_mode=True).run('Task')

    def test_schema_version_one_is_migrated_without_losing_sessions(self):
        agent = CodingAgent(Client({'content': 'done'}), self.runtime(), store=self.store)
        agent.run('Task')
        self.store.db.execute('PRAGMA user_version=1')
        self.store.db.close()
        self.store = SessionStore(self.root / '.agent/state.sqlite3')
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 2)
        self.assertEqual(self.store.get(agent.session_id).final_answer, 'done')
        self.assertIsNone(self.store.plan(agent.session_id))

    def test_corrupted_saved_plan_blocks_resume(self):
        first = CodingAgent(Client({'tool_calls': [submit()]}), self.runtime(), store=self.store, plan_mode=True)
        with self.assertRaises(PlanApprovalDeferred):
            first.run('Task')
        self.store.db.execute('UPDATE plans SET data=? WHERE session_id=?', ('{}', first.session_id))
        with self.assertRaises(SessionError):
            CodingAgent(Client(), self.runtime('auto'), store=self.store).resume(first.session_id)

    def test_interactive_recovery_retries_and_continues_same_run(self):
        runtime = self.runtime('auto')
        execute = runtime.execute
        def die(name, arguments):
            execute(name, arguments)
            raise KeyboardInterrupt
        runtime.execute = die
        first = CodingAgent(Client({'tool_calls': [write()]}), runtime, store=self.store)
        with self.assertRaises(KeyboardInterrupt):
            first.run('Task')
        decision = Mock(return_value='retry')
        self.assertEqual(CodingAgent(Client({'content': 'done'}), self.runtime('auto'), store=self.store,
                                     resolve_uncertain=decision).resume(first.session_id), 'done')
        decision.assert_called_once()
        self.assertEqual(self.store.get(first.session_id).rounds, 2)

    def test_interactive_recovery_skip_abort_and_applied(self):
        for choice in ('skip', 'abort', 'applied'):
            runtime = self.runtime('auto')
            execute = runtime.execute
            def die(name, arguments):
                execute(name, arguments)
                raise KeyboardInterrupt
            runtime.execute = die
            first = CodingAgent(Client({'tool_calls': [write()]}), runtime, store=self.store)
            with self.assertRaises(KeyboardInterrupt):
                first.run('Task')
            resumed = self.runtime('auto')
            agent = CodingAgent(Client({'content': 'done'}), resumed, store=self.store,
                                resolve_uncertain=lambda item: choice, verify_recovery=lambda item: verify_applied(resumed, item))
            if choice == 'abort':
                with self.assertRaises(SessionError):
                    agent.resume(first.session_id)
                self.assertEqual(self.store.get(first.session_id).status, 'aborted')
            else:
                self.assertEqual(agent.resume(first.session_id), 'done')
                content = self.store.calls(first.session_id)[0]['result']['content']
                self.assertIn('RECOVERY_SKIPPED' if choice == 'skip' else 'RECOVERY_CONFIRMED', content)


class PlanValidationTests(unittest.TestCase):
    def test_demo_never_silently_falls_back_to_scripted_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'Real model client required'):
                run_plan_demo(root / 'workspace', root / '.agent/state.sqlite3')
            self.assertFalse((root / 'workspace').exists())
            with self.assertRaisesRegex(ValueError, 'Offline demo cannot'):
                run_plan_demo(root / 'workspace', root / '.agent/state.sqlite3', offline=True, client=PlanDemoClient())

    def test_invalid_fields_steps_tools_and_criteria_are_rejected(self):
        cases = [None, {}, {**DEMO_PLAN, 'extra': 1}, {**DEMO_PLAN, 'steps': []},
                 {**DEMO_PLAN, 'steps': [{'description': 'x', 'tools': ['unknown']}]},
                 {**DEMO_PLAN, 'done_criteria': []}, {**DEMO_PLAN, 'goal': ' '},
                 {**DEMO_PLAN, 'risks': ['x' * 4001]}, {**DEMO_PLAN, 'steps': [{'description': 'x', 'tools': 'Bash'}]},
                 {**DEMO_PLAN, 'risks': ['x' * 4000] * 30}]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(PlanError):
                validate_plan(value)


class TerminalTests(unittest.TestCase):
    setUp = PlanningTests.setUp
    tearDown = PlanningTests.tearDown
    runtime = PlanningTests.runtime

    def test_plan_prompt_yes_no_empty_eof_and_no_terminal(self):
        for response, result in (('y', True), ('n', False), ('', None), (EOFError(), None)):
            with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=response if isinstance(response, Exception) else None,
                    return_value=response), redirect_stdout(io.StringIO()) as output:
                if result is None:
                    with self.assertRaises(PlanApprovalDeferred):
                        confirm_plan(DEMO_PLAN)
                else:
                    self.assertIs(confirm_plan(DEMO_PLAN), result)
            self.assertIn('Критерии готовности', output.getvalue())
        with patch('sys.stdin.isatty', return_value=False), redirect_stdout(io.StringIO()), self.assertRaises(PlanApprovalDeferred):
            confirm_plan(DEMO_PLAN)

    def test_session_selection_does_not_require_typing_ids(self):
        identifiers = [self.store.create({'workspace': str(self.workspace)}, [{'role': 'user', 'content': 'Task'}], 12, 120000) for _ in range(2)]
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=['bad', '2']), redirect_stdout(io.StringIO()):
            self.assertEqual(choose_session(self.store, None), identifiers[0])
        with patch('sys.stdin.isatty', return_value=False), self.assertRaises(SessionError):
            choose_session(self.store, None)
        self.assertEqual(choose_session(self.store, identifiers[0]), identifiers[0])

    def test_conflict_menu_shows_file_and_choices(self):
        item = {'key': '1:0', 'call': write()}
        for value, decision in (('1', 'retry'), ('2', 'skip'), ('3', 'abort'), ('4', 'applied'), ('0', None)):
            with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', return_value=value), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(confirm_conflict(self.runtime(), item, lambda item: True), decision)
            self.assertIn('original', output.getvalue())
            self.assertIn('proposed', output.getvalue())
        with patch('sys.stdin.isatty', return_value=False), patch('builtins.input') as prompt:
            self.assertIsNone(confirm_conflict(self.runtime(), item))
        prompt.assert_not_called()

    def test_noninteractive_cli_resume_without_id_fails_explicitly(self):
        with patch('sys.argv', ['hw-ai-agent', 'resume', '--state-db', str(self.store.path)]), \
             patch('sys.stdin.isatty', return_value=False), redirect_stderr(io.StringIO()) as output:
            self.assertEqual(main(), 1)
        self.assertIn('SESSION_ID', output.getvalue())

    def test_cli_run_and_demo_planning_never_preflight_bash(self):
        for action in ('plan', 'run', 'demo'):
            argv = ['hw-ai-agent', action, '--provider', 'ollama', '--model', 'fake-model',
                    '--workspace', str(self.workspace if action != 'demo' else self.root / 'new-demo'),
                    '--state-db', str(self.store.path), '--mode', 'auto']
            if action != 'plan':
                argv.append('--plan')
            if action != 'demo':
                argv.append('Task')
            client = Client({'tool_calls': [submit()]})
            with patch('sys.argv', argv), patch('hw_ai_agent.cli.OpenAICompatibleClient', return_value=client), \
                 patch('hw_ai_agent.cli.doctor', return_value=True) as doctor, \
                 patch('hw_ai_agent.cli.confirm_plan', return_value=False), patch.object(ToolRuntime, 'bash') as bash, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 1)
            bash.assert_not_called()
            self.assertEqual(len(client.requests), 1)
            if action == 'demo':
                self.assertEqual(doctor.call_args.kwargs, {'check_sandbox': False})
            self.assertEqual((self.workspace / 'order.py').read_text(), 'original')


@unittest.skipUnless(os.getenv('REQUIRE_SANDBOX_TESTS') == '1', 'Requires native sandbox')
class PlanDemoTests(unittest.TestCase):
    def test_cli_plan_demo_and_resume_without_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); db = root / '.agent/state.sqlite3'; workspace = root / 'workspace'
            with patch('sys.argv', ['hw-ai-agent', 'plan-demo', '--offline', '--mode', 'auto', '--workspace', str(workspace), '--state-db', str(db)]), \
                 patch('sys.stdin.isatty', return_value=False), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 130)
            with SessionStore(db) as store:
                saved = store.list_sessions()[0]
                self.assertEqual(saved['status'], 'waiting_plan_approval')
                self.assertEqual((workspace / 'order.py').read_text(), 'def calculate_total(prices, discount_percent):\n    return sum(prices) - discount_percent\n')
            with patch('sys.argv', ['hw-ai-agent', 'resume', '--state-db', str(db)]), patch('sys.stdin.isatty', return_value=True), \
                 patch('builtins.input', return_value='y'), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(), 0)
            self.assertIn('DEMO PASS', output.getvalue())
            with SessionStore(db) as store:
                self.assertEqual(store.list_sessions()[0]['rounds'], 4)
                self.assertEqual(store.plan(saved['id'])['state'], 'approved')

    def test_default_uses_model_client_and_resume_keeps_backend_and_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); db = root / '.agent/state.sqlite3'; workspace = root / 'workspace'
            settings = Settings('ollama', 'example-model', 'http://localhost:11434/v1', 'ollama')
            clients = []
            def factory(chosen, **kwargs):
                c = PlanDemoClient()
                c.settings, c.max_tokens = chosen, kwargs['max_tokens']
                c.complete = Mock(wraps=c.complete)
                clients.append(c)
                return c
            argv = ['hw-ai-agent', 'plan-demo', '--workspace', str(workspace), '--state-db', str(db),
                    '--mode', 'auto', '--max-tokens', '1472', '--max-rounds', '9']
            with patch('sys.argv', argv), patch('hw_ai_agent.cli.Settings.from_env', return_value=settings) as config, \
                 patch('hw_ai_agent.cli.OpenAICompatibleClient', side_effect=factory) as model, \
                 patch('hw_ai_agent.cli.doctor', return_value=True) as doctor, patch('sys.stdin.isatty', return_value=False), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 130)
            model.assert_called_once()
            config.assert_called_once_with(None, None)
            self.assertEqual(doctor.call_args.kwargs, {'check_sandbox': False})
            with SessionStore(db) as store:
                state = store.get(store.list_sessions()[0]['id'])
                self.assertEqual(state.metadata['provider'], 'ollama')
                self.assertEqual(state.metadata['model'], 'example-model')
                self.assertEqual(state.metadata['max_tokens'], 1472)
                self.assertEqual(state.max_rounds, 9)
                self.assertFalse(state.metadata['demo']['offline'])
            with patch('sys.argv', ['hw-ai-agent', 'resume', state.id, '--state-db', str(db)]), \
                 patch('hw_ai_agent.cli.Settings.from_env', return_value=settings), \
                 patch('hw_ai_agent.cli.OpenAICompatibleClient', side_effect=factory) as restored_model, \
                 patch('hw_ai_agent.cli.PlanDemoClient', side_effect=AssertionError('No scripted fallback')), \
                 patch('sys.stdin.isatty', return_value=True), patch('builtins.input', return_value='y'), \
                 redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(), 0)
            self.assertIn('DEMO PASS', output.getvalue())
            self.assertEqual(restored_model.call_args.kwargs['max_tokens'], 1472)
            self.assertEqual(len(clients), 2)
            report = json.loads((workspace / 'demo-report.json').read_text())
            self.assertEqual(report['provider'], 'ollama')
            self.assertEqual(report['model'], 'example-model')
            self.assertTrue(report['success'])

    def test_offline_skips_model_configuration_even_with_custom_token_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); db = root / '.agent/state.sqlite3'; workspace = root / 'workspace'
            argv = ['hw-ai-agent', 'plan-demo', '--offline', '--workspace', str(workspace), '--state-db', str(db),
                    '--mode', 'auto', '--max-tokens', '4096']
            with patch('sys.argv', argv), patch('hw_ai_agent.cli.OpenAICompatibleClient') as model, \
                 patch('hw_ai_agent.cli.load_env') as env, patch('sys.stdin.isatty', return_value=False), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 130)
            model.assert_not_called()
            env.assert_not_called()
            with patch('sys.argv', ['hw-ai-agent', 'resume', '--state-db', str(db)]), \
                 patch('hw_ai_agent.cli.OpenAICompatibleClient') as model, patch('sys.stdin.isatty', return_value=True), \
                 patch('builtins.input', return_value='y'), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(), 0)
            model.assert_not_called()
            self.assertIn('DEMO PASS', output.getvalue())
