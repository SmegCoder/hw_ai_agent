from __future__ import annotations

import io
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hw_ai_agent import cli
from hw_ai_agent.config import (ConfigurationError, LAUNCH_OPTIONS, START_COMMANDS,
                                apply_launch_defaults, load_env, option_value,
                                parse_env, update_env)
from hw_ai_agent.session import SessionStore


class ConfigCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = self.root / '.env'
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def invoke(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, 'argv', ['hw-ai-agent', *map(str, arguments)]), redirect_stdout(out), redirect_stderr(err):
            result = cli.main()
        return result, out.getvalue(), err.getvalue()

    def test_set_creates_file_without_model_key_or_session(self):
        with patch.object(cli, 'OpenAICompatibleClient') as model, patch.object(cli, 'SessionStore') as store:
            status, out, _ = self.invoke('set', '--env-file', self.env, '--provider', 'ollama', '--max-rounds', '30')
        self.assertEqual(status, 0)
        self.assertEqual(parse_env(self.env.read_text()), {'AGENT_PROVIDER': 'ollama', 'AGENT_MAX_ROUNDS': '30'})
        self.assertEqual(stat.S_IMODE(self.env.stat().st_mode), 0o600)
        self.assertIn('AGENT_MAX_ROUNDS', out)
        model.assert_not_called(); store.assert_not_called()

    def test_set_preserves_secrets_comments_crlf_and_other_models(self):
        original = b'# comment\r\nOPENROUTER_API_KEY="example-private-key"\r\nOPENROUTER_MODEL=old-router\r\nAGENT_PROVIDER=openrouter\r\nAGENT_MAX_ROUNDS=12\r\nAGENT_MAX_ROUNDS=13\r\n'
        self.env.write_bytes(original)
        status, out, err = self.invoke('set', '--env-file', self.env, '--provider', 'ollama', '--model', 'local-model', '--max-rounds', '40')
        self.assertEqual(status, 0, err)
        data = self.env.read_bytes()
        self.assertIn(b'# comment\r\nOPENROUTER_API_KEY="example-private-key"\r\nOPENROUTER_MODEL=old-router\r\n', data)
        self.assertEqual(data.count(b'AGENT_MAX_ROUNDS='), 1)
        self.assertEqual(parse_env(data.decode())['OLLAMA_MODEL'], 'local-model')
        self.assertNotIn('example-private-key', out + err)

    def test_set_model_uses_current_provider_without_requiring_key(self):
        self.env.write_text('AGENT_PROVIDER=openai\nOPENAI_MODEL=old\n')
        self.assertEqual(self.invoke('set', '--env-file', self.env, '--model', 'new-model')[0], 0)
        self.assertEqual(parse_env(self.env.read_text())['OPENAI_MODEL'], 'new-model')

    def test_set_model_respects_exported_provider(self):
        self.env.write_text('AGENT_PROVIDER=openai\n')
        os.environ['AGENT_PROVIDER'] = 'ollama'
        self.invoke('set', '--env-file', self.env, '--model', 'local-model')
        self.assertEqual(parse_env(self.env.read_text())['OLLAMA_MODEL'], 'local-model')

    def test_set_all_launch_options_and_boolean_off_switches(self):
        status, _, err = self.invoke('set', '--env-file', self.env,
            '--max-rounds', 31, '--max-tokens', 4096, '--max-tool-calls', 90,
            '--max-side-effects', 0, '--max-result-bytes', 20000, '--mode', 'auto',
            '--approvals', 'deny', '--workspace', self.root / 'project',
            '--state-db', self.root / 'state.sqlite3', '--policy', self.root / 'policy.json',
            '--plan', '--tasks', '--journal')
        self.assertEqual(status, 0, err)
        values = parse_env(self.env.read_text())
        self.assertEqual(set(values), {option.env for option in LAUNCH_OPTIONS.values()})
        self.assertEqual(values['AGENT_TASKS'], 'true')
        self.invoke('set', '--env-file', self.env, '--no-plan', '--no-tasks', '--no-journal')
        values = parse_env(self.env.read_text())
        for key in ('AGENT_PLAN', 'AGENT_TASKS', 'AGENT_JOURNAL'):
            self.assertEqual(values[key], 'false')

    def test_invalid_set_leaves_file_unchanged(self):
        original = b'OPENROUTER_API_KEY=example-private-key\n'
        self.env.write_bytes(original)
        for args in (('--max-rounds', '0'), ('--max-tokens', '-1'), ('--max-side-effects', '-1'), ('--model', 'bad\nKEY=bad')):
            with self.subTest(args=args):
                status, out, err = self.invoke('set', '--env-file', self.env, *args)
                self.assertEqual(status, 1)
                self.assertEqual(self.env.read_bytes(), original)
                self.assertNotIn('example-private-key', out + err)

    def test_set_without_options_does_not_create_file(self):
        self.assertEqual(self.invoke('set', '--env-file', self.env)[0], 1)
        self.assertFalse(self.env.exists())

    def test_set_refuses_symlink_and_hardlink(self):
        target = self.root / 'private'
        target.write_bytes(b'KEY=preserved\n')
        self.env.symlink_to(target)
        self.assertEqual(self.invoke('set', '--env-file', self.env, '--provider', 'ollama')[0], 1)
        self.assertEqual(target.read_bytes(), b'KEY=preserved\n')
        self.env.unlink(); os.link(target, self.env)
        self.assertEqual(self.invoke('set', '--env-file', self.env, '--provider', 'ollama')[0], 1)
        self.assertEqual(target.read_bytes(), b'KEY=preserved\n')

    def test_interrupted_atomic_replace_preserves_original(self):
        original = b'KEY=preserved\n'
        self.env.write_bytes(original)
        with patch('hw_ai_agent.config.os.replace', side_effect=OSError('replace failed')):
            with self.assertRaises(OSError): update_env(self.env, {'AGENT_MODE': 'auto'})
        self.assertEqual(self.env.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [self.env])

    def test_concurrent_change_before_replace_is_preserved(self):
        self.env.write_bytes(b'KEY=original\n')
        fsync = os.fsync
        def change(fd):
            self.env.write_bytes(b'KEY=external-change\n')
            fsync(fd)
        with patch('hw_ai_agent.config.os.fsync', side_effect=change):
            with self.assertRaises(ConfigurationError): update_env(self.env, {'AGENT_MODE': 'auto'})
        self.assertEqual(self.env.read_bytes(), b'KEY=external-change\n')

    def test_missing_file_appearing_during_update_is_not_overwritten(self):
        fsync = os.fsync
        def create(fd):
            self.env.write_bytes(b'KEY=external\n')
            fsync(fd)
        with patch('hw_ai_agent.config.os.fsync', side_effect=create):
            with self.assertRaises(ConfigurationError): update_env(self.env, {'AGENT_MODE': 'auto'})
        self.assertEqual(self.env.read_bytes(), b'KEY=external\n')

    def test_env_defaults_all_start_commands_and_original_fallbacks(self):
        for action in START_COMMANDS:
            args = SimpleNamespace(action=action, **{name: None for name in LAUNCH_OPTIONS})
            apply_launch_defaults(args)
            expected = 30 if action == 'tasks-demo' else 20 if action in {'workflow-demo', 'plan-demo', 'rollback-demo'} else 12
            self.assertEqual(args.max_rounds, expected)
            with patch.dict(os.environ, {'AGENT_MAX_ROUNDS': '44', 'AGENT_MAX_TOKENS': '4000', 'AGENT_MAX_TOOL_CALLS': '70', 'AGENT_MAX_SIDE_EFFECTS': '0', 'AGENT_MAX_RESULT_BYTES': '10000', 'AGENT_MODE': 'auto'}):
                args = SimpleNamespace(action=action, **{name: None for name in LAUNCH_OPTIONS})
                apply_launch_defaults(args)
                self.assertEqual((args.max_rounds, args.max_tokens, args.max_tool_calls, args.max_side_effects, args.max_result_bytes, args.mode), (44, 4000, 70, 0, 10000, 'auto'))

    def test_file_then_exported_environment_then_cli_precedence(self):
        self.env.write_text('AGENT_MAX_ROUNDS=20\nAGENT_MODE=safe\n')
        os.environ['AGENT_MAX_ROUNDS'] = '30'
        load_env(self.env)
        args = SimpleNamespace(action='run', max_rounds=None, mode=None)
        apply_launch_defaults(args)
        self.assertEqual(args.max_rounds, 30)
        args = SimpleNamespace(action='run', max_rounds=40, mode='auto')
        apply_launch_defaults(args)
        self.assertEqual((args.max_rounds, args.mode), (40, 'auto'))

    def test_invalid_env_values_and_zero_allowed_budgets(self):
        for name, value in (('max_rounds', '0'), ('max_tokens', 'abc'), ('max_tool_calls', '-1'), ('mode', 'unsafe'), ('journal', 'maybe'), ('workspace', '')):
            with self.subTest(name=name), self.assertRaises(ConfigurationError): option_value(name, value)
        for name in ('max_tool_calls', 'max_side_effects', 'max_result_bytes'):
            self.assertEqual(option_value(name, '0'), 0)

    def test_boolean_cli_overrides_env_and_inherent_demo_modes(self):
        with patch.dict(os.environ, {'AGENT_PLAN': 'true', 'AGENT_TASKS': 'true', 'AGENT_JOURNAL': 'false'}):
            args = SimpleNamespace(action='run', plan=False, tasks=False, journal=None)
            apply_launch_defaults(args)
            self.assertFalse(args.plan); self.assertFalse(args.tasks); self.assertFalse(args.journal)
            args = SimpleNamespace(action='plan', plan=None, tasks=False)
            apply_launch_defaults(args); self.assertTrue(args.plan)
            args = SimpleNamespace(action='rollback-demo', journal=None)
            apply_launch_defaults(args); self.assertTrue(args.journal)

    def test_demo_does_not_use_configured_user_workspace(self):
        os.environ['AGENT_WORKSPACE'] = str(self.root / 'user-project')
        for action in START_COMMANDS:
            args = SimpleNamespace(action=action, workspace=None)
            apply_launch_defaults(args)
            self.assertEqual(args.workspace, Path(os.environ['AGENT_WORKSPACE']) if action in {'run', 'plan'} else None)

    def test_setup_populates_common_defaults_and_private_permissions(self):
        self.assertEqual(self.invoke('setup', '--provider', 'ollama', '--env-file', self.env)[0], 0)
        values = parse_env(self.env.read_text())
        self.assertEqual(values['AGENT_MAX_ROUNDS'], '12')
        self.assertEqual(values['AGENT_MAX_RESULT_BYTES'], '1000000')
        self.assertEqual(stat.S_IMODE(self.env.stat().st_mode), 0o600)

    def test_env_limits_saved_and_resume_ignores_new_limits(self):
        workspace, db = self.root / 'demo', self.root / 'sessions.sqlite3'
        self.env.write_text(f'AGENT_MAX_ROUNDS=19\nAGENT_MAX_TOKENS=1600\nAGENT_MAX_TOOL_CALLS=50\nAGENT_MAX_SIDE_EFFECTS=10\nAGENT_MAX_RESULT_BYTES=20000\nAGENT_MODE=auto\nAGENT_STATE_DB={db}\n')
        status, _, err = self.invoke('plan-demo', '--offline', '--env-file', self.env, '--workspace', workspace, '--stop-after-tools', 1)
        self.assertEqual(status, 130, err)
        with SessionStore(db) as store:
            identifier = store.list_sessions()[0]['id']
            before = store.get(identifier)
            self.assertEqual(before.max_rounds, 19)
            self.assertEqual(before.metadata['max_tokens'], 1600)
            budget = store.budget(identifier)
            self.assertEqual(budget['limits']['tool_calls'], 50)
        self.env.write_text(f'AGENT_MAX_ROUNDS=bad\nAGENT_MAX_TOKENS=bad\nAGENT_MODE=bad\nAGENT_STATE_DB={db}\n')
        # A separate process would have an empty environment again.
        os.environ.clear()
        status, _, err = self.invoke('resume', identifier, '--env-file', self.env, '--stop-after-tools', 1)
        self.assertEqual(status, 130, err)
        with SessionStore(db) as store:
            after = store.get(identifier)
            self.assertEqual(after.max_rounds, 19)
            self.assertEqual(after.metadata['max_tokens'], 1600)
            self.assertEqual(store.budget(identifier)['limits'], budget['limits'])

    def test_invalid_env_fails_before_api_and_workspace_creation(self):
        self.env.write_text('AGENT_MAX_ROUNDS=wrong\n')
        workspace = self.root / 'new-demo'
        with patch.object(cli, 'OpenAICompatibleClient') as model:
            status, _, err = self.invoke('tasks-demo', '--env-file', self.env, '--workspace', workspace)
        self.assertEqual(status, 1)
        self.assertIn('AGENT_MAX_ROUNDS', err)
        self.assertFalse(workspace.exists()); model.assert_not_called()

    def test_real_run_cli_overrides_file_and_saves_effective_limits(self):
        workspace, db = self.root / 'project', self.root / 'sessions.sqlite3'
        workspace.mkdir()
        self.env.write_text(f'AGENT_PROVIDER=ollama\nOLLAMA_MODEL=file-model\nAGENT_WORKSPACE={workspace}\nAGENT_STATE_DB={db}\nAGENT_MAX_ROUNDS=bad\nAGENT_MAX_TOKENS=1800\nAGENT_MAX_TOOL_CALLS=44\nAGENT_MAX_SIDE_EFFECTS=5\nAGENT_MAX_RESULT_BYTES=20000\nAGENT_MODE=safe\nAGENT_PLAN=true\nAGENT_TASKS=true\nAGENT_JOURNAL=true\n')
        class Client:
            usage = []
            def __init__(self, settings, **kwargs):
                self.settings, self.max_tokens = settings, kwargs['max_tokens']
            def complete(self, messages, tools): return {'content': 'done'}
        with patch.object(cli, 'OpenAICompatibleClient', side_effect=Client) as factory:
            status, _, err = self.invoke('run', '--env-file', self.env,
                '--model', 'cli-model', '--max-rounds', 35, '--max-tokens', 2400,
                '--mode', 'auto', '--no-plan', '--no-tasks', '--no-journal', 'Inspect')
        self.assertEqual(status, 0, err)
        self.assertEqual(factory.call_args.args[0].model, 'cli-model')
        self.assertEqual(factory.call_args.kwargs['max_tokens'], 2400)
        with SessionStore(db) as store:
            identifier = store.list_sessions()[0]['id']
            session = store.get(identifier)
            self.assertEqual(session.max_rounds, 35)
            self.assertEqual(session.metadata['execution_mode'], 'auto')
            for name in ('plan_mode', 'task_mode', 'journal_mode'):
                self.assertFalse(session.metadata[name])
            self.assertEqual(session.metadata['max_tokens'], 2400)
            self.assertEqual(store.budget(identifier)['limits']['tool_calls'], 44)

    def test_offline_reads_limits_without_loading_api_keys(self):
        self.env.write_text('OPENROUTER_API_KEY=example-private-key\nAGENT_MAX_ROUNDS=9\n')
        with patch.object(cli, 'OpenAICompatibleClient') as model:
            status, _, err = self.invoke('plan-demo', '--offline', '--env-file', self.env,
                '--workspace', self.root / 'demo', '--state-db', self.root / 'db', '--stop-after-tools', 1)
        self.assertEqual(status, 130, err)
        self.assertEqual(os.environ['AGENT_MAX_ROUNDS'], '9')
        self.assertNotIn('OPENROUTER_API_KEY', os.environ)
        model.assert_not_called()

    def test_help_does_not_read_env_or_update_settings(self):
        with patch.object(cli, 'load_env') as load, patch.object(cli, 'set_options') as update:
            status, out, _ = self.invoke('help', 'set')
        self.assertEqual(status, 0)
        self.assertIn('AGENT_MAX_ROUNDS', out)
        load.assert_not_called(); update.assert_not_called()


if __name__ == '__main__':
    unittest.main()
