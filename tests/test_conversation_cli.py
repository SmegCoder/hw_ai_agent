from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import chdir, redirect_stderr, redirect_stdout
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_fixtures import PLAN, tool_call
from hw_ai_agent import cli
from hw_ai_agent.agent import DEFAULT_SYSTEM_PROMPT
from hw_ai_agent.conversation import Conversation
from hw_ai_agent.journal import ChangeJournal
from hw_ai_agent.model_catalog import Model
from hw_ai_agent.pricing import Price, estimate_tokens
from hw_ai_agent.session import SessionStore, encoded
from hw_ai_agent.tools import ToolRuntime


class ConversationCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.file = self.workspace / "value.py"
        self.file.write_text("value = 0\n")
        self.db = self.root / "state.db"
        self.env_file = self.root / ".env"
        self.responses = []
        self.requests = []
        self.output_limits = []
        self.window = 20_000

        environment = patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        test = self

        class Client:
            billing_enabled = True

            def __init__(self, settings, *, max_tokens, **kwargs):
                self.settings, self.max_tokens = settings, max_tokens
                self.usage = []

            def complete(self, messages, tools):
                test.requests.append([(item["role"], item.get("content")) for item in messages])
                test.output_limits.append(self.max_tokens)
                response = test.responses.pop(0)
                if isinstance(response, BaseException):
                    raise response
                if isinstance(response, tuple):
                    response, usage = response
                else:
                    usage = {"total_tokens": 10}
                self.usage.append(usage)
                return response

        for temporary_patch in (
            patch.object(cli, "OpenAICompatibleClient", Client),
            patch.object(cli, "catalog_for", side_effect=self.catalog),
            patch("sys.stdin.isatty", return_value=True),
        ):
            temporary_patch.start()
            self.addCleanup(temporary_patch.stop)

    def catalog(self, args, provider):
        model = Model("test/model", Price("0", "0"), context=self.window, tools=True)
        return SimpleNamespace(find=lambda name: model if name == model.id else None,
                               source="test", updated_at="test", status="test", warning=None)

    def invoke(self, action, *arguments, inputs=(), tty_output=False):
        out, err = io.StringIO(), io.StringIO()
        argv = ["hw-ai-agent", action, "--env-file", str(self.env_file), "--state-db", str(self.db), *map(str, arguments)]
        with patch.object(sys, "argv", argv), patch("builtins.input", side_effect=inputs), \
             patch.object(out, "isatty", return_value=tty_output), \
             redirect_stdout(out), redirect_stderr(err):
            status = cli.main()
        return status, out.getvalue(), err.getvalue()

    def start(self, *inputs):
        return self.invoke("chat", "--provider", "openrouter", "--model", "test/model", "--mode", "auto",
                           "--max-tokens", 20, "--workspace", self.workspace, inputs=inputs)

    def conversation_id(self):
        with SessionStore(self.db) as store:
            return store.db.execute("SELECT id FROM conversations").fetchone()["id"]

    @staticmethod
    def write(value, path="value.py"):
        return {"tool_calls": [tool_call("write", "write_file", {"path": path, "content": f"value = {value}\n"})]}

    def test_rollback_latest_turn_then_continue_with_history_and_totals(self):
        self.responses = [self.write(1), {"content": "Первое изменение."},
                          self.write(2), {"content": "Второе изменение."},
                          ({"content": "После отката."}, {})]
        code, out, err = self.start("Первая правка", "Вторая правка", "/changes", "/rollback",
                                    "Что осталось?", "/tokens", "/stop")
        self.assertEqual(code, 0, err)
        self.assertIn("+value = 2", out)
        self.assertIn("Запрос 2 отменён", out)
        self.assertIn("Токены за диалог · ≈", out)
        self.assertNotIn("Без отчёта провайдера", out)
        self.assertNotIn("расход известен частично", out)
        self.assertEqual(out.count("Окно модели · 20 000 токенов"), 1)
        self.assertLess(out.index("Окно модели"), out.index("Первое изменение."))
        self.assertIn("▶ Изменения · /changes", out)
        self.assertEqual(self.file.read_text(), "value = 1\n")
        self.assertIn(("assistant", "Первое изменение."), self.requests[-1])
        self.assertTrue(any(content and "отменил правки запросов 2" in content for _, content in self.requests[-1]))
        identifier = self.conversation_id()
        self.responses = [{"content": "Память сохранилась."}]
        code, out, err = self.invoke("chat", "--resume", identifier, inputs=("Напомни", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn(("assistant", "После отката."), self.requests[-1])
        self.assertIn("Диалог завершён", out)
        with SessionStore(self.db) as store:
            turns = Conversation(store, identifier).turns()
            self.assertEqual([turn.state.status for turn in turns], ["completed", "rolled_back", "completed", "completed"])
            self.assertEqual(Conversation.tokens(turns)[0], 50)
            self.assertEqual(len({turn.state.id for turn in turns}), 4)
            self.assertNotIn("test-key", store.db.execute("SELECT configuration FROM conversations").fetchone()[0])
        code, _, err = self.invoke("chat", "--resume", identifier)
        self.assertEqual(code, 1)
        self.assertIn("окончательно завершён", err)

    def test_ctrl_c_saves_request_and_resume_keeps_its_journal(self):
        self.responses = [self.write(1), KeyboardInterrupt()]
        code, out, err = self.start("Исправь", "/end")
        self.assertEqual(code, 130, err)
        self.assertIn("Диалог на паузе", out)
        identifier = self.conversation_id()
        self.responses = [{"content": "Готово."}, {"content": "После отката."}]
        code, out, err = self.invoke("chat", "--resume", identifier,
                                     inputs=("Новый запрос слишком рано", "/resume", "/rollback", "Продолжим", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn("Есть незавершённый запрос", out)
        self.assertIn("Токены за диалог · ≈", out)
        self.assertNotIn("расход известен частично", out)
        self.assertEqual(self.file.read_text(), "value = 0\n")
        self.assertNotIn(("user", "Новый запрос слишком рано"), self.requests[-1])

    def test_chat_settings_persist_and_resumed_turn_uses_its_checkpoint(self):
        self.responses = [self.write(1), KeyboardInterrupt()]
        self.assertEqual(self.start("Первая правка")[0], 130)
        identifier = self.conversation_id()
        self.responses = [{"content": "Продолжено."}, self.write(2), {"content": "Запись отклонена."}]
        code, out, err = self.invoke("chat", "--resume", identifier,
            inputs=("/set", "/set approvals invalid", "/set mode safe", "/set approvals deny",
                    "/set max-rounds 4", "/set max-tool-calls 1", "/set max-side-effects 1",
                    "/set max-tokens 50", "/settings", "/resume", "Вторая правка", "/stop"))
        self.assertEqual(code, 0, err)
        self.assertIn("Формат · /set", out)
        self.assertIn("допустимы ask, deny", out)
        self.assertIn("approvals · deny", out)
        self.assertIn("max-tool-calls · 1", out)
        self.assertIn("APPROVAL_REQUIRED", out)
        self.assertEqual(self.output_limits, [20, 20, 20, 50, 50])
        self.assertEqual(self.file.read_text(), "value = 1\n")
        self.responses = [self.write(3), {"content": "Записано с подтверждением."}]
        code, out, err = self.invoke("chat", "--resume", identifier,
                                     inputs=("/settings", "/set approvals ask", "Третья правка", "y", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn("mode · safe", out)
        self.assertIn("max-tokens · 50", out)
        self.assertEqual(self.file.read_text(), "value = 3\n")
        with SessionStore(self.db) as store:
            first, second, third = Conversation(store, identifier).turns()
            self.assertEqual(first.state.metadata["execution_mode"], "auto")
            self.assertEqual(second.state.metadata["approvals"], "deny")
            self.assertEqual(third.state.metadata["approvals"], "ask")
            self.assertEqual(store.budget(second.state.id)["limits"],
                             {"model_requests": 4, "tool_calls": 1, "side_effects": 1, "result_bytes": 1_000_000})

    def test_chat_progress_persists_and_applies_to_the_pending_request(self):
        self.responses = [self.write(1), KeyboardInterrupt()]
        code, out, err = self.start("/set progress true", "Первая правка")
        self.assertEqual(code, 130, err)
        self.assertIn("Файл записан · value.py", out)
        identifier = self.conversation_id()
        self.responses = [{"content": "Продолжено."}]
        code, out, err = self.invoke("chat", "--resume", identifier, inputs=("/settings", "/resume", "/stop"))
        self.assertEqual(code, 0, err)
        self.assertIn("progress · true", out)
        self.assertRegex(out, r"\d{2}:\d{2}  Запрос к модели · 3")

        self.responses = [{"content": "Готово."}]
        code, out, err = self.invoke("chat", "--resume", identifier, "--no-progress",
                                     inputs=("/settings", "Ещё запрос", "/stop"))
        self.assertEqual(code, 0, err)
        self.assertIn("progress · false", out)
        self.assertIn("Агент работает…", out)
        self.assertNotIn("Запрос к модели", out)
        with SessionStore(self.db) as store:
            self.assertTrue(Conversation(store, identifier).configuration()["progress"])

    def test_chat_handles_surrogate_text_without_rewriting_invalid_tool_arguments(self):
        call = tool_call("broken-write", "write_file", {"path": "value.py", "content": "value = 1"})
        call["function"]["arguments"] = call["function"]["arguments"].replace("value = 1", "value = 1\udcd0")
        self.responses = [
            {"tool_calls": [call], "reasoning_details": [{"text": "Проверка\udcd0"}]},
            {"content": "Ответ\udcd0"},
            {"content": "Продолжение ✓"},
        ]
        code, out, err = self.start("Запрос\udcd0", "Продолжи", "/end")
        self.assertEqual(code, 0, err)
        self.assertIn(r"Ответ\udcd0", out)
        self.assertIn("Продолжение ✓", out)
        self.assertEqual(self.file.read_text(), "value = 0\n")
        self.assertIn(("user", r"Запрос\udcd0"), self.requests[0])
        self.assertIn(("assistant", r"Ответ\udcd0"), self.requests[-1])
        with SessionStore(self.db) as store:
            first, second = Conversation(store, self.conversation_id()).turns()
            self.assertEqual([first.state.status, second.state.status], ["completed", "completed"])
            rejected = store.calls(first.state.id)[0]
            self.assertEqual(rejected["state"], "failed")
            self.assertIn("valid UTF-8 text", rejected["result"]["content"])

    def test_old_chat_without_progress_uses_the_env_default(self):
        self.assertEqual(self.start("/stop")[0], 0)
        identifier = self.conversation_id()
        with SessionStore(self.db) as store:
            configuration = Conversation(store, identifier).configuration()
            del configuration["progress"]
            store.db.execute("UPDATE conversations SET configuration=? WHERE id=?", (encoded(configuration), identifier))
        self.env_file.write_text("AGENT_PROGRESS=true\n")
        self.responses = [{"content": "Готово."}]
        code, out, err = self.invoke("chat", "--resume", identifier, inputs=("/settings", "Запрос", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn("progress · true", out)
        self.assertIn("Запрос к модели", out)

    def test_interrupted_rollback_must_finish_before_next_prompt(self):
        writes = self.write(1)["tool_calls"] + [
            tool_call("write_other", "write_file", {"path": "z.py", "content": "value = 1\n"})
        ]
        (self.workspace / "z.py").write_text("value = 0\n")
        self.responses = [{"tool_calls": writes}, {"content": "Два файла."}]
        original = ChangeJournal._rollback

        def interrupted(journal):
            def pause(_path):
                raise KeyboardInterrupt
            return original(journal, after_file=pause)

        with patch.object(ChangeJournal, "_rollback", interrupted):
            code, _, err = self.start("Измени оба файла", "/rollback")
        self.assertEqual(code, 130, err)
        self.assertEqual(self.file.read_text(), "value = 0\n")
        self.assertEqual((self.workspace / "z.py").read_text(), "value = 1\n")
        code, out, err = self.invoke("chat", "--resume", self.conversation_id(),
                                     inputs=("Новая задача", "/resume", "/end", "/rollback", "/end"))
        self.assertEqual(code, 0, err)
        self.assertIn("Сначала завершите прерванный откат", out)
        self.assertEqual((self.workspace / "z.py").read_text(), "value = 0\n")
        self.assertEqual(len(self.requests), 2)

    def test_context_warning_and_refusal_happen_before_model_request(self):
        prompt = "Первый вопрос"
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        self.window = estimate_tokens(messages, ToolRuntime(self.workspace).tool_schemas) + 240
        self.responses = [{"content": "Ответ."}]
        oversized = "Очень большой запрос " * 1_000
        code, out, err = self.start(prompt, oversized, "/tokens", "/end")
        self.assertEqual(code, 0, err)
        self.assertIn("Контекст близок к пределу", err)
        self.assertIn("Запрос не отправлен", out)
        self.assertEqual(len(self.requests), 1)
        with SessionStore(self.db) as store:
            turns = Conversation(store, self.conversation_id()).turns()
            self.assertEqual(turns[-1].state.status, "context_limit")
            self.assertEqual(turns[-1].state.rounds, 0)

    def test_run_without_workspace_selects_a_numbered_folder(self):
        for name in ("wok", "fold"):
            (self.workspace / name).mkdir()
        (self.workspace / "outside").symlink_to(self.root, target_is_directory=True)
        self.responses = [{"content": "Готово."}]
        with chdir(self.root), patch.dict(os.environ, {"AGENT_WORKSPACE": str(self.root / "other")}):
            code, out, err = self.invoke("run", "--model", "test/model", "Проверь проект", inputs=("2",))
        self.assertEqual(code, 0, err)
        self.assertIn("1. fold", out)
        self.assertIn("2. wok", out)
        self.assertNotIn("outside", out)
        with SessionStore(self.db) as store:
            session = store.get(store.list_sessions()[0]["id"])
            self.assertEqual(Path(session.metadata["workspace"]), (self.workspace / "wok").resolve())

    def test_enter_creates_dated_folders_for_each_new_plan(self):
        with chdir(self.root), patch("hw_ai_agent.terminal.date") as clock:
            clock.today.return_value = date(2026, 11, 24)
            for name in ("2026-11-24", "2026-11-24-2"):
                self.responses = [{"tool_calls": [tool_call("plan", "submit_plan", PLAN)]}, {"content": "Готово."}]
                code, out, err = self.invoke("plan", "--model", "test/model", "Проверь проект", inputs=("", "y"))
                self.assertEqual(code, 0, err)
                self.assertIn(f"Создана папка · workspace/{name}", out)
                self.assertTrue((self.workspace / name).is_dir())
        with SessionStore(self.db) as store:
            self.assertEqual({Path(session["workspace"]).name for session in store.list_sessions()},
                             {"2026-11-24", "2026-11-24-2"})

    def test_chat_can_create_a_named_workspace_and_reopen_it_without_menu(self):
        (self.workspace / "wok").mkdir()
        with chdir(self.root):
            code, out, err = self.invoke("chat", "--model", "test/model",
                                         inputs=("n", "../outside", "n", "wok", "n", "Новый проект", "/stop"))
            self.assertEqual(code, 0, err)
            self.assertIn("одно имя папки без пути", out)
            self.assertIn("Это имя уже занято", out)
            self.assertIn("Создана папка · workspace/Новый проект", out)
            code, out, err = self.invoke("chat", "--resume", self.conversation_id(), inputs=("/end",))
        self.assertEqual(code, 0, err)
        self.assertNotIn("Рабочая папка · workspace", out)
        self.assertIn("Новый проект", out)
        self.assertFalse((self.root / "outside").exists())

    def test_workspace_selection_can_be_cancelled_before_starting(self):
        with chdir(self.root):
            code, _, err = self.invoke("run", "--model", "test/model", "Проверь проект",
                                       inputs=(KeyboardInterrupt(),))
        self.assertEqual(code, 130)
        self.assertIn("Выбор рабочей папки отменён", err)
        self.assertFalse(self.db.exists())
        self.assertEqual(self.requests, [])

    def test_single_prompt_output_is_compact_and_verbose_is_opt_in(self):
        options = ("--provider", "openrouter", "--model", "test/model", "--mode", "auto",
                   "--workspace", self.workspace, "--max-tokens", 20)
        self.responses = [self.write(1), {"content": "Готово."}, self.write(2), {"content": "Ещё готово."}]
        code, out, err = self.invoke("run", *options, "Исправь")
        self.assertEqual(code, 0, err)
        self.assertIn("✓  value.py записан", out)
        self.assertIn("Тесты не запускались", out)
        self.assertNotIn("Вызов write_file", out)
        self.assertNotIn("Бюджет:", out)
        self.assertNotIn("Запрос к модели", out)
        code, out, err = self.invoke("run", *options, "--verbose", "Ещё исправь")
        self.assertEqual(code, 0, err)
        self.assertIn("Вызов write_file", out)
        self.assertIn("Оценка запроса:", out)
        with SessionStore(self.db) as store:
            self.assertEqual(len(store.list_sessions()), 2)
            self.assertEqual(store.db.execute("SELECT count(*) FROM conversations").fetchone()[0], 0)


    def test_result_shows_verified_files_test_count_and_saved_diff(self):
        options = ("--provider", "openrouter", "--model", "test/model", "--mode", "auto", "--journal",
                   "--workspace", self.workspace, "--max-tokens", 20)
        self.env_file.write_text("AGENT_PROGRESS=true\n")
        check = {"tool_calls": [tool_call("tests", "Bash", {"command": "python3 -m unittest discover -v"})]}
        self.responses = [self.write(1), check, {"content": "Готово."}]
        output = "exit_code=0\nstdout:\n\nstderr:\nRan 1 test in 0.001s\n\nOK"
        with patch.object(ToolRuntime, "bash", return_value=output):
            code, out, err = self.invoke("run", *options, "Исправь и проверь")

        self.assertEqual(code, 0, err)
        self.assertIn("agi run", out)
        self.assertIn("OpenRouter · test/model", out)
        self.assertIn("Задача выполнена", out)
        self.assertIn("✓  value.py обновлён", out)
        self.assertIn("✓  Тесты прошли · 1 из 1", out)
        for line in ("Запрос к модели · 1", "Файл записан · value.py", "Проверка · python3", "Тесты прошли\n"):
            self.assertIn(line, out)
        self.assertIn("▶ Изменения · agi changes", out)
        self.assertIn("--state-db", out)
        with SessionStore(self.db) as store:
            session_id = store.list_sessions()[0]["id"]
        code, out, err = self.invoke("changes", session_id)
        self.assertEqual(code, 0, err)
        self.assertIn("+value = 1", out)

    def test_safe_approval_shows_color_before_asking_and_sanitizes_file_content(self):
        self.file.write_text("-- old\x1b[2J")
        self.responses = [self.write(1), {"content": "Готово."}]
        options = ("--model", "test/model", "--mode", "safe", "--journal", "--workspace", self.workspace,
                   "--max-tokens", 20, "Исправь")

        def approve(prompt):
            self.assertIn("[y/N]", prompt)
            output = sys.stdout.getvalue()
            self.assertIn("\x1b[31m--- old\\x1b[2J\x1b[0m", output)
            self.assertIn("\x1b[32m+value = 1\x1b[0m", output)
            self.assertIn("\\ No newline at end of file", output)
            self.assertNotIn("\x1b[2J", output)
            return "y"

        with patch.dict(os.environ, {"TERM": "xterm"}):
            code, _, err = self.invoke("run", *options, inputs=approve, tty_output=True)
            self.assertEqual(code, 0, err)
            self.assertEqual(self.file.read_text(), "value = 1\n")
            with SessionStore(self.db) as store:
                identifier = store.list_sessions()[0]["id"]
            code, out, err = self.invoke("changes", identifier, tty_output=True)
            self.assertEqual(code, 0, err)
            self.assertIn("\x1b[32m+value = 1\x1b[0m", out)
            with patch.dict(os.environ, {"NO_COLOR": "1"}):
                code, out, err = self.invoke("changes", identifier, tty_output=True)
            self.assertEqual(code, 0, err)
            self.assertNotIn("\x1b[", out)
            self.assertIn("+value = 1", out)

    def test_failed_tests_do_not_get_a_success_label(self):
        options = ("--provider", "openrouter", "--model", "test/model", "--mode", "auto",
                   "--workspace", self.workspace, "--max-tokens", 20)
        check = {"tool_calls": [tool_call("tests", "Bash", {"command": "python -m unittest discover"})]}
        self.responses = [self.write(1), check, {"content": "Прочтите результат проверки."}]
        output = "exit_code=1\nstdout:\n\nstderr:\nRan 1 test in 0.001s\n\nFAILED (failures=1)"
        with patch.object(ToolRuntime, "bash", return_value=output):
            code, out, err = self.invoke("run", *options, "Исправь и проверь")
        self.assertEqual(code, 0, err)
        self.assertIn("Успех тестов не подтверждён", out)
        self.assertNotIn("Задача выполнена", out)
        self.assertNotIn("▶ Изменения", out)


if __name__ == "__main__":
    unittest.main()
