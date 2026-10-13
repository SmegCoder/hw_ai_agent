import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from http.client import IncompleteRead

from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.budget import BudgetLimits
from hw_ai_agent.client import ModelClientError, OpenAICompatibleClient
from hw_ai_agent.config import Settings
from hw_ai_agent.planning import PlanError
from hw_ai_agent.policy import Policy
from hw_ai_agent.session import SessionStore
from hw_ai_agent.tasks import validate_task_plan
from hw_ai_agent.tasks_demo import TASK_PLAN
from hw_ai_agent.tools import ToolError, ToolRuntime


def call(identifier, name, **arguments):
    return {'id': identifier, 'function': {'name': name, 'arguments': json.dumps(arguments)}}


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.messages = []

    def complete(self, messages, tools):
        self.messages = copy.deepcopy(messages)
        return next(self.responses)


class RuntimeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        (self.workspace / 'order.py').write_text('original')

    def runtime(self, **kwargs):
        return ToolRuntime(self.workspace, **kwargs)

    def test_policy_file_missing_defaults_does_not_grant_other_tools(self):
        path = self.root / 'policy.json'
        path.write_text(json.dumps({'directories': {'allow': ['.']}, 'rules': [
            {'tool': 'read_file', 'path': 'order.py', 'action': 'allow'}]}))
        runtime = self.runtime(policy=Policy.load(path), mode='auto')
        self.assertEqual(runtime.execute('read_file', {'path': 'order.py'}), 'original')
        for name, args in [('write_file', {'path': 'order.py', 'content': 'bad'}),
                           ('Bash', {'command': 'pwd'})]:
            with self.subTest(name=name), patch.object(runtime, 'bash') as process:
                with self.assertRaises(ToolError):
                    runtime.execute(name, args)
                process.assert_not_called()
        self.assertEqual((self.workspace / 'order.py').read_text(), 'original')

    def test_policy_file_missing_directories_grants_no_workspace_access(self):
        path = self.root / 'policy.json'
        path.write_text(json.dumps({'defaults': {'read_file': 'allow'}}))
        with self.assertRaises(ToolError):
            self.runtime(policy=Policy.load(path)).execute('read_file', {'path': 'order.py'})

    def test_policy_change_during_approval_is_rechecked(self):
        runtime = self.runtime()
        def approve(request):
            runtime.policy = Policy.from_dict({'defaults': {'write_file': 'deny'}})
            return True
        runtime.approve = approve
        with self.assertRaises(ToolError):
            runtime.execute('write_file', {'path': 'order.py', 'content': 'bad'})
        self.assertEqual((self.workspace / 'order.py').read_text(), 'original')

    def test_bash_cwd_alias_change_during_approval_is_rejected(self):
        for name in ('first', 'second'):
            (self.workspace / name).mkdir()
        alias = self.workspace / 'alias'
        alias.symlink_to(self.workspace / 'first', target_is_directory=True)
        runtime = self.runtime()
        def approve(request):
            alias.unlink()
            alias.symlink_to(self.workspace / 'second', target_is_directory=True)
            return True
        runtime.approve = approve
        with patch.object(runtime, 'bash') as process:
            with self.assertRaises(ToolError):
                runtime.execute('Bash', {'command': 'pwd', 'cwd': 'alias'})
            process.assert_not_called()

    def test_unittest_option_paths_are_validated_before_starting_bash(self):
        runtime = self.runtime(mode='auto', policy=Policy.from_dict({
            'directories': {'allow': ['.'], 'deny': ['private']}}))
        commands = ['python -m unittest discover --start-directory=private',
                    'python -m unittest discover -sprivate',
                    'python -m unittest discover -s../outside',
                    'python -m unittest discover --top-level-directory=/usr',
                    'python -m unittest discover -t../outside',
                    'python -m unittest private.test_order',
                    'python -m unittest private.test_order.py',
                    'python -m unittest discover --start-directory=private.tests',
                    'python -m unittest discover --start-dir=private']
        for command in commands:
            with self.subTest(command=command), patch.object(runtime, 'bash') as process:
                with self.assertRaises(ToolError):
                    runtime.execute('Bash', {'command': command})
                process.assert_not_called()

    def test_safe_unittest_options_and_test_patterns_remain_usable(self):
        runtime = self.runtime(mode='auto')
        for command in ['python3 -m unittest -v',
                        'python -m unittest discover -s . -p "test_*.py" -v',
                        'python -m unittest discover --start-directory=. --pattern=test_*.py',
                        'python -m unittest test_order.OrderTest -k discount']:
            with self.subTest(command=command), patch.object(runtime, 'bash', return_value='ok'):
                self.assertEqual(runtime.execute('Bash', {'command': command}), 'ok')

    def test_invalid_unicode_argument_returns_tool_error_without_writing(self):
        runtime = self.runtime(mode='auto')
        client = Client({'tool_calls': [call('bad', 'write_file', path='order.py', content='\ud800')]},
                        {'tool_calls': [call('good', 'read_file', path='order.py')]}, {'content': 'done'})
        with SessionStore(self.root / 'state.db') as store:
            agent = CodingAgent(client, runtime, store=store)
            self.assertEqual(agent.run('Task'), 'done')
            self.assertEqual(store.calls(agent.session_id)[0]['state'], 'failed')
        self.assertEqual((self.workspace / 'order.py').read_text(), 'original')

    def test_runtime_limits_reject_nonfinite_or_wrong_types(self):
        for options in [{'command_timeout': float('nan')}, {'command_timeout': float('inf')},
                        {'max_file_chars': 2.5}, {'max_output_chars': True}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.runtime(**options)

    def test_error_prefix_in_file_content_is_not_a_failed_read(self):
        (self.workspace / 'log.txt').write_text('ERROR: this is project data')
        client = Client({'tool_calls': [call('read', 'read_file', path='log.txt')]}, {'content': 'done'})
        with SessionStore(self.root / 'state.db') as store:
            agent = CodingAgent(client, self.runtime(), store=store)
            agent.run('Read the log')
            self.assertEqual(store.calls(agent.session_id)[0]['state'], 'completed')

    def test_cached_truncated_error_stays_failed_after_resume(self):
        bad = call('bad', 'read_file', path='missing.txt')
        limits = BudgetLimits(result_bytes=4)
        client = Client({'tool_calls': [bad, call('tail', 'read_file', path='order.py')]})
        def stop(event):
            if event['event'] == 'tool_result':
                raise KeyboardInterrupt
        with SessionStore(self.root / 'state.db') as store, patch('hw_ai_agent.agent.output_ceiling', return_value=1):
            agent = CodingAgent(client, self.runtime(), store=store, budget_limits=limits, on_event=stop)
            with self.assertRaises(KeyboardInterrupt):
                agent.run('Read')
            self.assertEqual(store.calls(agent.session_id)[0]['result']['content'], '~')
            # Replaying the same ID after restart must retain its failed status
            # even though the one-byte result no longer includes "ERROR:".
            resumed = CodingAgent(Client({'tool_calls': [bad]}, {'content': 'done'}), self.runtime(), store=store)
            resumed.resume(agent.session_id)
            self.assertEqual([i['state'] for i in store.calls(agent.session_id)], ['failed', 'completed', 'failed'])


class PlanRegressionTests(unittest.TestCase):
    def test_malformed_verification_kind_is_a_recoverable_plan_error(self):
        for kind in [[], {}, None, 42]:
            plan = copy.deepcopy(TASK_PLAN)
            plan['steps'][0]['verification']['kind'] = kind
            with self.subTest(kind=kind), self.assertRaises(PlanError):
                validate_task_plan(plan)

    def test_agent_can_correct_malformed_plan_and_wait_for_approval(self):
        from hw_ai_agent.planning import PlanApprovalDeferred
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            workspace.mkdir()
            malformed = copy.deepcopy(TASK_PLAN)
            malformed['steps'][0]['verification']['kind'] = []
            client = Client({'tool_calls': [call('bad', 'submit_plan', **malformed)]},
                            {'tool_calls': [call('good', 'submit_plan', **TASK_PLAN)]})
            with SessionStore(root / 'state.db') as store:
                agent = CodingAgent(client, ToolRuntime(workspace), store=store, task_mode=True)
                with self.assertRaises(PlanApprovalDeferred):
                    agent.run('Plan')
                self.assertEqual([i['state'] for i in store.calls(agent.session_id)], ['failed', 'completed'])
                self.assertEqual(store.plan(agent.session_id)['state'], 'awaiting_approval')
                self.assertEqual(list(workspace.iterdir()), [])


class JournalPreviewRegressionTests(unittest.TestCase):
    def test_empty_file_creation_and_deletion_are_visible_before_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            workspace.mkdir()
            (workspace / 'deleted.txt').write_bytes(b'')
            with SessionStore(root / 'state.db') as store:
                client = Client({'content': 'done'})
                agent = CodingAgent(client, ToolRuntime(workspace), store=store, journal_mode=True)
                agent.run('Inspect')
                (workspace / 'created.txt').write_bytes(b'')
                (workspace / 'deleted.txt').unlink()
                view = agent.journal.recovery_view({'key': 'observed'})
                self.assertTrue(view['changed'])
                self.assertIn('created.txt', view['diff'])
                self.assertIn('deleted.txt', view['diff'])
                self.assertNotEqual(view['diff'], '(no file changes)')


class ClientRegressionTests(unittest.TestCase):
    def client(self):
        return OpenAICompatibleClient(Settings('openrouter', 'model', 'https://example.invalid/v1', 'secret'))

    def test_http_error_response_is_closed(self):
        body = io.BytesIO(b'error body')
        error = HTTPError('https://example.invalid', 401, 'bad', {}, body)
        client = self.client()
        with patch.object(client.opener, 'open', side_effect=error), self.assertRaises(ModelClientError):
            client.complete([], [])
        self.assertTrue(body.closed)

    def test_incomplete_http_response_becomes_a_sanitized_client_error(self):
        client = self.client()
        with patch.object(client.opener, 'open', side_effect=IncompleteRead(b'secret')), self.assertRaises(ModelClientError) as caught:
            client.complete([], [])
        self.assertNotIn('secret', str(caught.exception))
